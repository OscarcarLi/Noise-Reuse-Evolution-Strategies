"""Noise-Reuse Evolution Strategies.
For each particle, applies the same noise for the entire trajectory until resets."""
import functools
from typing import Mapping, Sequence, Tuple, Any

import haiku as hk
import jax
import jax.numpy as jnp
from src.utils import profile
from src.utils import summary
from src.utils import tree_utils
from src.utils import common
from src.outer_trainers import gradient_learner
from src.task_parallelization import truncated_step
import chex


PRNGKey = jax.Array
MetaParams = Any
UnrollState = Any
TruncatedUnrollState = Any

import flax.struct
import ipdb

# this is different from vector_sample_perturbations
# as we don't need the positive and negatively perturbed thetas
# (theta, key, std) # only keys are different over multiple samples
sample_multiple_perturbations = jax.jit(jax.vmap(common.sample_perturbations, in_axes=(None, 0, None)))


@flax.struct.dataclass
class TruncatedNRESAttributes(gradient_learner.GradientEstimatorState):
  pos_unroll_states: TruncatedUnrollState
  neg_unroll_states: TruncatedUnrollState
  epsilons: jax.Array

# previously named TruncatedESSharedNoise
class TruncatedNRES(gradient_learner.GradientEstimator):
    # does multiple particles already

  def __init__(
      self,
      truncated_step: truncated_step.TruncatedStep,
      unroll_length: int = 20,
      std: float = 0.01,
      burn_in_length: int = 0,
      jitted: bool = True,
  ):
    """Initializer.

    Args:
      truncated_step: class containing functions for initializing and
        progressing a inner-training state.
      unroll_length: length of the unroll
        unroll_length.
      std: standard deviation for ES.
      burn_in_length: how many steps to run unroll starting from the inner states
        returned from truncated_step.init_step_state to ensure all the states are
        obtained by using the current theta (+ epsilon)
    """
    self.truncated_step = truncated_step
    self.unroll_length = unroll_length
    self.std = std
    self.burn_in_length = burn_in_length
    self._total_env_steps_used = 0
    self.jitted = jitted
    if self.jitted:
      self.nres_unroll = jax.jit(self._nres_unroll)
    else:
      self.nres_unroll = self._nres_unroll

  def grad_est_name(self):
    return ("TruncatedNRES_"
            f"N={self.truncated_step.num_tasks},K=T,W={self.unroll_length},sigma={self.std}")

  def task_name(self):
    return self.truncated_step.task_name()

  @property
  def total_env_steps_used(self,):
    return self._total_env_steps_used

  @profile.wrap()
  def init_worker_state(self, worker_weights: gradient_learner.WorkerWeights,
                        key: PRNGKey) -> TruncatedNRESAttributes:
    key1, key2, key3 = jax.random.split(key, num=3)
    pos_unroll_states = self.truncated_step.init_step_state(
        worker_weights.theta,
        worker_weights.outer_state,
        jax.random.split(key1, self.truncated_step.num_tasks),
        theta_is_vector=False)
    neg_unroll_states = self.truncated_step.init_step_state(
        worker_weights.theta,
        worker_weights.outer_state,
        jax.random.split(key1,  self.truncated_step.num_tasks),
        theta_is_vector=False)

    # we use sample_perturbations instead of vector_sample_perturbations
    # as we don't need the positively/negatively perturbed thetas
    keys = jax.random.split(key2, self.truncated_step.num_tasks)
    epsilons = sample_multiple_perturbations(
      worker_weights.theta, keys, self.std)
    truncated_es_sharednoise_attributes = TruncatedNRESAttributes(
      pos_unroll_states=pos_unroll_states,
      neg_unroll_states=neg_unroll_states,
      epsilons=epsilons)
    
    # a burn-in period to ensure the theta's have its states all unrolled by itself
    if self.burn_in_length > 0:
      rng = hk.PRNGSequence(key3)  # type: ignore
      for _ in range(self.burn_in_length):
        data = self.truncated_step.get_batch()
        curr_key = next(rng)
        truncated_es_sharednoise_attributes, _, _, _ =\
          self.nres_unroll(
              worker_weights.theta,
              truncated_es_sharednoise_attributes,
              curr_key,
              data,
              worker_weights.outer_state)
      self._total_env_steps_used += (
        int(jnp.sum(truncated_es_sharednoise_attributes.pos_unroll_states.inner_step)) + \
        int(jnp.sum(truncated_es_sharednoise_attributes.neg_unroll_states.inner_step))
      )
    
    return truncated_es_sharednoise_attributes

  @profile.wrap()
  def compute_gradient_estimate(
      self,
      worker_weights,
      key: PRNGKey,
      state: TruncatedNRESAttributes, # this is the same state returned by init_worker_state
      with_summary=False,
  ) -> Tuple[gradient_learner.GradientEstimatorOut, Mapping[str, jax.Array]]:

    # because we have a for loop we let haiku manages the key
    rng = hk.PRNGSequence(key)

    theta = worker_weights.theta

    # sum the gradients over both
    # 1. the unroll_length
    # 2. the number of particles
    # for the step when the particle resets,
    # we exclude that step's contributed gradient
    loss_sum = jnp.array(0.0)
    g_sum = jax.tree_util.tree_map(jnp.zeros_like, theta)
    # cannot use jnp.zeros(shape=1) for total_count (this will be a length 1 array which is undesired)
    total_count = jnp.array(0)
    

    for i in range(self.unroll_length):
      data = self.truncated_step.get_batch()
      curr_key = next(rng)
      
      state, loss_sum_step, g_sum_step, count =\
        self.nres_unroll(
            theta,
            state,
            curr_key,
            data,
            worker_weights.outer_state)

      self._total_env_steps_used += 2 * self.truncated_step.num_tasks

      loss_sum += loss_sum_step
      g_sum = jax.tree_util.tree_map(lambda x, y: x + y, g_sum, g_sum_step)
      total_count += count

    # average over both particle and number of unroll step (this makes sure we are optimizing the average loss over all time steps)
    # because each particle over one step contributes both a pos and neg loss
    # we divide 2 times total_count
    mean_loss = loss_sum / (2 * total_count)
    g = jax.tree_util.tree_map(lambda x: x / total_count, g_sum)

    # unroll_info = gradient_learner.UnrollInfo(
    #     loss=p_ys.loss,
    #     iteration=p_ys.iteration,
    #     task_param=p_ys.task_param,
    #     is_done=p_ys.is_done)

    output = gradient_learner.GradientEstimatorOut(
        mean_loss=mean_loss,
        grad=g,
        unroll_state=state,
        unroll_info=None)

    return output, {}


  def _nres_unroll(
    self,
    theta: MetaParams, # there is a single copy of theta
    state: TruncatedNRESAttributes, 
    key: chex.PRNGKey,
    data: Any, # single batch of data to be used for both pos and neg unroll
    outer_state: Any) -> Tuple[TruncatedNRESAttributes, jax.Array, MetaParams, jax.Array]:
      # returns new_state, 

    # unpack the current state
    pos_unroll_states, neg_unroll_states, epsilons = \
      state.pos_unroll_states, state.neg_unroll_states, state.epsilons
    
    # compute the perturbation for this unroll step
    # there is one perturbed pos/neg theta for each trajectory
    pos_perturbed_thetas = jax.tree_util.tree_map(lambda a,b: jnp.expand_dims(a, 0) + b, theta, epsilons)
    neg_perturbed_thetas = jax.tree_util.tree_map(lambda a,b: jnp.expand_dims(a, 0) - b, theta, epsilons)

    key1, key2 = jax.random.split(key)
    pos_unroll_states, pos_outs = \
      self.truncated_step.unroll_step(
        theta=pos_perturbed_thetas,
        unroll_state=pos_unroll_states,
        key_list=jax.random.split(key1, self.truncated_step.num_tasks),
        data=data,
        outer_state=outer_state,
        theta_is_vector=True)
    neg_unroll_states, neg_outs = \
      self.truncated_step.unroll_step(
        theta=neg_perturbed_thetas,
        unroll_state=neg_unroll_states,
        key_list=jax.random.split(key1, self.truncated_step.num_tasks),
        # using the same key as pos ensures when resetting we get the same initial inner state and
        # and also ensures we get the same loss evaluation (if it takes randomness)
        data=data, # also use the same data
        outer_state=outer_state,
        theta_is_vector=True)
        
    # keep track of sum of losses for logging
    # pos_outs.loss is an array of losses (one for each trajectory/particle)
    # need to exclude the 0 entries when the trajectory has resetted and a loss of 0 is returned. 
    loss_sum_step = jnp.sum(pos_outs.loss * pos_outs.mask + neg_outs.loss * neg_outs.mask)
        
    # we set the multiplicative weight for the particle that resets to 0
    multiplier = ((pos_outs.loss - neg_outs.loss) * pos_outs.mask) * 1 / (2 * self.std ** 2) 
    # g_sum_step is equal to the sum of gradient estimates for all particles
    # which has non trivial gradient estimates (this excludes the particles)
    # that has reset in this unroll.                            
    g_sum_step = jax.tree_util.tree_map(lambda eps:
            jnp.sum( # sum over all particles (each particle is working on one trajectory)
              eps * jnp.reshape(multiplier, [multiplier.shape[0]] + [1]*(len(eps.shape)-1)),
              axis=0),
                              epsilons)
    # count keeps track of how many gradient estimates are summed in this unroll
    count = jnp.sum(pos_outs.mask)

    # for the particle that resets, we sample a new epsilon
    keys = jax.random.split(key2, self.truncated_step.num_tasks)
    new_epsilons = sample_multiple_perturbations(theta, keys, self.std)
    # replace epsilon of the trajectory that has finished with a new epsilon
    def update_eps(eps, new_eps):
      reshape_isdone = jnp.reshape(pos_outs.is_done,
                                    [self.truncated_step.num_tasks] + [1] * (len(eps.shape)-1))
      return eps * (1 - reshape_isdone) + new_eps * (reshape_isdone)
    epsilons = jax.tree_util.tree_map(lambda eps, new_eps: update_eps(eps, new_eps), epsilons, new_epsilons)

    return (
      TruncatedNRESAttributes(
              pos_unroll_states=pos_unroll_states,
              neg_unroll_states=neg_unroll_states,
              epsilons=epsilons),
      loss_sum_step,
      g_sum_step,
      count)
