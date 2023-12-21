"""Generalized PES with the same noise for every K steps, output an estimator
every W steps (unroll_length)"""
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
class TruncatedPESKAttributes(gradient_learner.GradientEstimatorState):
  pos_unroll_states: TruncatedUnrollState
  neg_unroll_states: TruncatedUnrollState
  current_epsilons: jax.Array # the epsilon already used before and added to accumulated_epsilons in past iteration.
  accumulated_epsilons: jax.Array

class TruncatedPESK(gradient_learner.GradientEstimator):
  """The generalized PES gradient estimator
  for each particle, it resamples a new noise every K steps, outputs a gradient
  estimate every W steps."""

  def __init__(
      self,
      truncated_step: truncated_step.TruncatedStep,
      unroll_length: int = 20,
      K: int = 1,
      std: float = 0.01,
      burn_in_length: int = 0,
      loss_normalize=False,
      jitted: bool = True,
  ):
    """Initializer.

    Args:
      truncated_step: class containing functions for initializing and
        progressing a inner-training state.
      unroll_length: truncation window size W for each worker. 
      K: how frequent to generate a new epsilon
      std: standard deviation for ES.
      burn_in_length: how many steps to run unroll starting from the inner states
        returned from truncated_step.init_step_state to ensure all the states are
        obtained by using the current theta (+ epsilon)
      loss_normalize: whether to normalize the gradient by the std of the loss
      jitted: boolean whether to jit the core computations
    """
    self.truncated_step = truncated_step
    self.unroll_length = unroll_length
    self.K = K
    self.std = std
    self.burn_in_length = burn_in_length
    self.loss_normalize = loss_normalize
    self._total_env_steps_used = 0
    self.jitted = jitted
    if self.jitted:
      self.pesk_unroll = jax.jit(self._pesk_unroll)
    else:
      self.pesk_unroll = self._pesk_unroll


  def grad_est_name(self):
    name_list = [
      "TruncatedPESK",
      f"_N={self.truncated_step.num_tasks},K={self.K},W={self.unroll_length},sigma={self.std}"
    ]
    if self.loss_normalize:
      name_list.insert(1, "loss_normalized")
    return "_".join(name_list)

  def task_name(self):
    return self.truncated_step.task_name()

  @property
  def total_env_steps_used(self,):
    return self._total_env_steps_used

  @profile.wrap()
  def init_worker_state(self, worker_weights: gradient_learner.WorkerWeights,
                        key: PRNGKey) -> TruncatedPESKAttributes:
    key1, key2, key3 = jax.random.split(key, num=3)
    # here we don't assume unroll_states is immutable
    pos_unroll_states = self.truncated_step.init_step_state(
        worker_weights.theta,
        worker_weights.outer_state,
        jax.random.split(key1, self.truncated_step.num_tasks),
        theta_is_vector=False)
    neg_unroll_states = self.truncated_step.init_step_state(
        worker_weights.theta,
        worker_weights.outer_state,
        jax.random.split(key1, self.truncated_step.num_tasks),
        theta_is_vector=False)

    # we use zeros as the starting point for accumulated PES
    zeros = jax.tree_util.tree_map(lambda x: jnp.zeros(shape=[self.truncated_step.num_tasks] + list(x.shape)), worker_weights.theta)
    keys = jax.random.split(key2, self.truncated_step.num_tasks)
    epsilons = sample_multiple_perturbations(
      worker_weights.theta, keys, self.std)
    truncated_pesk_attributes = TruncatedPESKAttributes(
      pos_unroll_states=pos_unroll_states,
      neg_unroll_states=neg_unroll_states,
      current_epsilons=epsilons,
      accumulated_epsilons=zeros)
    
    # a burn-in period to ensure the theta's have its states all unrolled by itself
    if self.burn_in_length > 0:
      rng = hk.PRNGSequence(key3)  # type: ignore
      for _ in range(self.burn_in_length):
        print(_)
        data = self.truncated_step.get_batch()
        curr_key = next(rng)
        (truncated_pesk_attributes, _, _, _), _ =\
          self.pesk_unroll(
              worker_weights.theta,
              truncated_pesk_attributes,
              curr_key,
              data,
              worker_weights.outer_state)
      self._total_env_steps_used += (
        int(jnp.sum(truncated_pesk_attributes.pos_unroll_states.inner_step)) + \
        int(jnp.sum(truncated_pesk_attributes.neg_unroll_states.inner_step))
      )
    
    return truncated_pesk_attributes

  @profile.wrap()
  def compute_gradient_estimate(
      self,
      worker_weights,
      key: PRNGKey,
      state: TruncatedPESKAttributes, # this is the same state returned by init_worker_state
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
    pos_list = []
    neg_list = []

    for i in range(self.unroll_length):
      data = self.truncated_step.get_batch()
      curr_key = next(rng)
      
      # import ipdb; ipdb.set_trace()
      (state, loss_sum_step, g_sum_step, count), (pos_loss, neg_loss) =\
        self.pesk_unroll(
            theta,
            state,
            curr_key,
            data,
            worker_weights.outer_state)
      self._total_env_steps_used += 2 * self.truncated_step.num_tasks
      pos_list.append(pos_loss)
      neg_list.append(neg_loss)
      # import ipdb; ipdb.set_trace()
      
      # if i in [0, 1]:
        # import ipdb; ipdb.set_trace()

      loss_sum += loss_sum_step
      g_sum = jax.tree_util.tree_map(lambda x, y: x + y, g_sum, g_sum_step)
      total_count += count

    # import ipdb; ipdb.set_trace()
    
    # average over both particle and number of unroll step (this makes sure we are optimizing the average loss over all time steps)
    # because each particle over one step contributes both a pos and neg loss
    # we divide 2 times total_count
    mean_loss = loss_sum / (2 * total_count)
    if self.loss_normalize:
      # there should be totally (2 * selfunroll_length, num_particles) losses
      avg_pos_list = jnp.mean(jnp.array(pos_list), axis=0) # shape (num_particles,)
      avg_neg_list = jnp.mean(jnp.array(neg_list), axis=0) # shape (num_particles,)
      all_losses = jnp.stack((avg_pos_list, avg_neg_list), axis=0) # shape (2, num_particles)
      loss_std = jnp.std(all_losses)
      g = jax.tree_util.tree_map(lambda x: x * self.std / loss_std / total_count, g_sum)
    else:
      loss_std = None
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

    if with_summary:
      if loss_std is not None:
        return output, {"mean||loss_std": loss_std}
    return output, {}


  def _pesk_unroll(
    self,
    theta: MetaParams, # there is a single copy of theta
    state: TruncatedPESKAttributes, 
    key: chex.PRNGKey,
    data: Any, # single batch of data to be used for both pos and neg unroll
    outer_state: Any) -> Tuple[Tuple[TruncatedPESKAttributes, jax.Array, MetaParams, jax.Array], Tuple[jax.Array, jax.Array]]:
      # returns new_state, 

    # unpack the current state
    pos_unroll_states, neg_unroll_states, current_epsilons, accumulated_epsilons = \
      state.pos_unroll_states, state.neg_unroll_states, state.current_epsilons, state.accumulated_epsilons

    key1, key2, key3 = jax.random.split(key, num=3)

    # import ipdb; ipdb.set_trace()

    # we need to figure out whether we need to have a newly sampled epsilon or not
    # here we want a boolean vector with truncated_step.num_tasks elements
    need_new_epsilons = (jnp.remainder(pos_unroll_states.inner_step, self.K) == 0)
    def update_noise(eps, new_eps):
      # whether to keep the original eps or the new_eps depending 
      # the boolean in need_new_epsilons for each trajectory
      # when the inner_step is divisble by K, get a new epsilon
      reshape_need_new = jnp.reshape(need_new_epsilons,
                                    [self.truncated_step.num_tasks] + [1] * (len(eps.shape)-1))
      return eps * (1 - reshape_need_new) + new_eps * (reshape_need_new)

    # we see whether to update the current_epsilon
    keys = jax.random.split(key1, self.truncated_step.num_tasks)
    new_epsilons = sample_multiple_perturbations(theta, keys, self.std)
    current_epsilons = jax.tree_util.tree_map(lambda eps, new_eps: update_noise(eps, new_eps),
                          current_epsilons, new_epsilons)
    # accumulate the new_epsilons for the trajectory that needs to refresh
    accumulated_epsilons = \
      jax.tree_util.tree_map(lambda eps, new_eps: update_noise(eps, new_eps),
        accumulated_epsilons, # don't need new epsilon
        tree_utils.tree_add(accumulated_epsilons, new_epsilons)) # need new epsilon
    
    # compute the perturbation for this unroll step
    # there is one perturbed pos/neg theta for each trajectory
    pos_perturbed_thetas = jax.tree_util.tree_map(lambda a,b: jnp.expand_dims(a, 0) + b, theta, current_epsilons)
    neg_perturbed_thetas = jax.tree_util.tree_map(lambda a,b: jnp.expand_dims(a, 0) - b, theta, current_epsilons)

    pos_unroll_states, pos_outs = \
      self.truncated_step.unroll_step(
        theta=pos_perturbed_thetas,
        unroll_state=pos_unroll_states,
        key_list=jax.random.split(key2, self.truncated_step.num_tasks),
        data=data,
        outer_state=outer_state,
        theta_is_vector=True)
    neg_unroll_states, neg_outs = \
      self.truncated_step.unroll_step(
        theta=neg_perturbed_thetas,
        unroll_state=neg_unroll_states,
        key_list=jax.random.split(key2, self.truncated_step.num_tasks),
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
    g_per_particle = jax.tree_util.tree_map(
      lambda eps: eps * jnp.reshape(multiplier, [multiplier.shape[0]] + [1]*(len(eps.shape)-1)),
      accumulated_epsilons)
    g_sum_step = jax.tree_util.tree_map(lambda x:
            jnp.sum( # sum over all particles (each particle is working on one trajectory)
              x,
              axis=0),
                              g_per_particle)
    # import ipdb; ipdb.set_trace()
    # count keeps track of how many gradient estimates are summed in this unroll
    count = jnp.sum(pos_outs.mask)

    ### if a particular trajectory has finished, we reset its current_epsilons and accumulated noise
    def reset_eps(eps):
      # for the trajectory that has resetted, we set its current_epsilons
      # and accumulated_epsilons to zero
      reshape_isdone = jnp.reshape(pos_outs.is_done,
                                    [self.truncated_step.num_tasks] + [1] * (len(eps.shape)-1))
      return eps * (1 - reshape_isdone)
    
    # import ipdb; ipdb.set_trace()
    # currently we set the epsilon to zero as in the next iteration with inner_step == 0,
    # we will sample a new current_epsilons
    current_epsilons = jax.tree_util.tree_map(reset_eps, current_epsilons)
    # replace accumulated epsilon of the trajectory that has finished with zero
    accumulated_epsilons = jax.tree_util.tree_map(reset_eps, accumulated_epsilons)

    return (
      TruncatedPESKAttributes(
              pos_unroll_states=pos_unroll_states,
              neg_unroll_states=neg_unroll_states,
              current_epsilons=current_epsilons,
              accumulated_epsilons=accumulated_epsilons),
      loss_sum_step,
      g_sum_step,
      count), (pos_outs.loss, neg_outs.loss)
