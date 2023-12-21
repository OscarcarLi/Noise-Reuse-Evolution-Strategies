"""Online directional gradient descent applied over truncations (same noise for each particle).
Based on the algorithm DODGE in Learning by Directional Gradient Descent
https://openreview.net/forum?id=5i7lJLuhTm
"""
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

def partition_state_into_float_nonfloat(state):
    partitions, unflattener = \
        tree_utils.partition([lambda k,v: jnp.asarray(v).dtype == jnp.float32 or jnp.asarray(v).dtype == jnp.float64], state, strict=False)
    return partitions, unflattener

@flax.struct.dataclass
class TruncatedForwardModeAttributes(gradient_learner.GradientEstimatorState):
  unroll_states: TruncatedUnrollState
  epsilons: jax.Array
  float_state_jvp_wepsilon: jax.Array

class TruncatedForwardModeSharedNoise(gradient_learner.GradientEstimator):
  """The Online version of directional gradient descent with shared noise
  over each episode."""

  def __init__(
      self,
      truncated_step: truncated_step.TruncatedStep,
      unroll_length: int = 20,
      burn_in_length: int = 0,
  ):
    """Initializer.

    Args:
      truncated_step: class containing functions for initializing and
        progressing a inner-training state.
      unroll_length: length of the unroll truncation window.
      burn_in_length: how many steps to run unroll starting from the inner states
        returned from truncated_step.init_step_state to ensure all the states are
        obtained by using the current theta (+ epsilon)
    """
    self.truncated_step = truncated_step
    self.unroll_length = unroll_length
    self.burn_in_length = burn_in_length
    self._total_env_steps_used = 0


  def grad_est_name(self):
    # there is no sigma parameter for TruncatedForwardMode
    return \
      ("TruncatedForwardModeSharedNoise"
      f"_N={self.truncated_step.num_tasks},K=T,W={self.unroll_length}")

  def task_name(self):
    return self.truncated_step.task_name()

  @property
  def total_env_steps_used(self,):
    return self._total_env_steps_used

  def theta_tile(self, theta):
    return jax.tree_util.tree_map(
      lambda eps: jnp.tile(eps, reps=[self.truncated_step.num_tasks] + [1 for i in eps.shape]),
      theta)

  @profile.wrap()
  def init_worker_state(self, worker_weights: gradient_learner.WorkerWeights,
                        key: PRNGKey) -> TruncatedForwardModeAttributes:
    key1, key2, key3 = jax.random.split(key, num=3)
    tiled_thetas = self.theta_tile(worker_weights.theta)

    def init_function(thetas):
        unroll_states = self.truncated_step.init_step_state(
            thetas,
            worker_weights.outer_state,
            jax.random.split(key1, self.truncated_step.num_tasks),
            theta_is_vector=True)
        (float_unrollstates, nonfloat_unrollstates), unflattener = \
            partition_state_into_float_nonfloat(unroll_states)
        return float_unrollstates, unroll_states

    # we use sample_perturbations instead of vector_sample_perturbations
    # as we don't need the positively/negatively perturbed thetas
    keys = jax.random.split(key2, self.truncated_step.num_tasks)
    epsilons = sample_multiple_perturbations(
      worker_weights.theta, keys, 1)
    
    # when jvp with_aux returns (primals_out, tangents_out, aux)
    _, float_state_jvp_wepsilon, unroll_states = \
          jax.jvp(init_function,
            primals=(tiled_thetas,),
            tangents=(epsilons,),
            has_aux=True)
    truncated_forwardmode_attributes = TruncatedForwardModeAttributes(
      unroll_states=unroll_states,
      epsilons=epsilons,
      float_state_jvp_wepsilon=float_state_jvp_wepsilon)
    
    # a burn-in period to ensure the theta's have its states all unrolled by itself
    if self.burn_in_length > 0:
      rng = hk.PRNGSequence(key3)  # type: ignore
      for _ in range(self.burn_in_length):
        data = self.truncated_step.get_batch()
        curr_key = next(rng)
        (truncated_forwardmode_attributes, _, _, _), _ =\
          self.directional_forwardmode_unroll(
              worker_weights.theta,
              truncated_forwardmode_attributes,
              curr_key,
              data,
              worker_weights.outer_state)
      self._total_env_steps_used += int(jnp.sum(truncated_forwardmode_attributes.unroll_states.inner_step))
    
    return truncated_forwardmode_attributes

  @profile.wrap()
  def compute_gradient_estimate(
      self,
      worker_weights,
      key: PRNGKey,
      state: TruncatedForwardModeAttributes, # this is the same state returned by init_worker_state
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
      
      (state, loss_sum_step, g_sum_step, count), multiplier =\
        self.directional_forwardmode_unroll(
            theta,
            state,
            curr_key,
            data,
            worker_weights.outer_state)

      self._total_env_steps_used += self.truncated_step.num_tasks

      # import ipdb; ipdb.set_trace()
      # print(jnp.max(multiplier), jnp.min(multiplier))
      loss_sum += loss_sum_step
      g_sum = jax.tree_util.tree_map(lambda x, y: x + y, g_sum, g_sum_step)
      total_count += count

    # average over both particle and number of unroll step (this makes sure we are optimizing the average loss over all time steps)
    # here each particle only contributes a single loss (because we are doing forward prop)
    # we divide by 1 times total_count
    mean_loss = loss_sum / total_count 
    g = jax.tree_util.tree_map(lambda x: x / total_count, g_sum)

    # if tree_utils.tree_norm(g) >= 1000:
    #   import ipdb; ipdb.set_trace()

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


  @functools.partial(
      jax.jit, static_argnums=(0,))
  def directional_forwardmode_unroll(
    self,
    theta: MetaParams, # there is a single copy of theta
    state: TruncatedForwardModeAttributes, 
    key: chex.PRNGKey,
    data: Any, # single batch of data to be used for both pos and neg unroll
    outer_state: Any
    ) -> Tuple[Tuple[TruncatedForwardModeAttributes, jax.Array, MetaParams, jax.Array], jax.Array]:
      # returns new_state, 
    
    tiled_thetas = self.theta_tile(theta) # repeated theta; one for each task
    key1, key2 = jax.random.split(key)

    # unpack the current state
    unroll_states, epsilons, float_state_jvp_wepsilon = \
      state.unroll_states, state.epsilons, state.float_state_jvp_wepsilon
    
    (float_unrollstates, nonfloat_unrollstates), unflattener = \
            partition_state_into_float_nonfloat(unroll_states)

    def transition_function(float_state, thetas):
        # assemble the complete state
        # here we use nonfloat_unrollstates defined outside this function
        reassembled_unrollstate = tree_utils.partition_unflatten(
          unflattener=unflattener, part_values=[float_state, nonfloat_unrollstates])  # type: ignore
        next_unrollstate, ys = self.truncated_step.unroll_step(
            theta=thetas,
            unroll_state=reassembled_unrollstate,
            key_list=jax.random.split(key1, self.truncated_step.num_tasks),
            data=data,
            outer_state=outer_state,
            theta_is_vector=True)
        (new_float_state, _), _ = partition_state_into_float_nonfloat(next_unrollstate)
        return (new_float_state, ys.loss), (next_unrollstate, ys)
    
    _, (float_state_jvp_wepsilon, loss_jvp_wepsilon), (next_unroll_states, outs) = \
      jax.jvp(transition_function,
              primals=(float_unrollstates, tiled_thetas),
              tangents=(float_state_jvp_wepsilon, epsilons),
              has_aux=True)

    # keep track of sum of losses for logging
    # pos_outs.loss is an array of losses (one for each trajectory/particle)
    # need to exclude the 0 entries when the trajectory has resetted yet a loss is still returned
    # because we overwrite the 0 loss with a meta loss (of the initialization) in lopt_truncated_step
    loss_sum_step = jnp.sum(outs.loss * outs.mask)
        
    # we set the multiplicative weight for the particle that resets to 0
    multiplier = loss_jvp_wepsilon * outs.mask 
    # g_sum_step is equal to the sum of gradient estimates for all particles
    # which has non trivial gradient estimates (this excludes the particles)
    # that has reset in this unroll.                            
    g_sum_step = jax.tree_util.tree_map(lambda eps:
            jnp.sum( # sum over all particles (each particle is working on one trajectory)
              eps * jnp.reshape(multiplier, [multiplier.shape[0]] + [1]*(len(eps.shape)-1)),
              axis=0),
                              epsilons)
    # count keeps track of how many gradient estimates are summed in this unroll
    count = jnp.sum(outs.mask)

    # for the particle that has reached its end, we sample a new epsilon
    keys = jax.random.split(key2, self.truncated_step.num_tasks)
    new_epsilons = sample_multiple_perturbations(theta, keys, 1)
    # replace epsilon of the trajectory that has finished with a new epsilon
    def update_eps(eps, new_eps):
      reshape_isdone = jnp.reshape(outs.is_done,
                                    [self.truncated_step.num_tasks] + [1] * (len(eps.shape)-1))
      return eps * (1 - reshape_isdone) + new_eps * (reshape_isdone)
    epsilons = jax.tree_util.tree_map(lambda eps, new_eps: update_eps(eps, new_eps), epsilons, new_epsilons)
    ### here we should technically consider also resetting the float_state_jvp_wepsilon
    # if a trajectory has reached its end. However, if the new state (in the next trajectory) doesn't
    # depend on the last state in the finished trajectory, the float_state_jvp_wepsilon
    # is already calculated to be zero when we run this function again next time.

    return (
      TruncatedForwardModeAttributes(
              unroll_states=next_unroll_states,
              epsilons=epsilons,
              float_state_jvp_wepsilon=float_state_jvp_wepsilon),
      loss_sum_step,
      g_sum_step,
      count), multiplier
