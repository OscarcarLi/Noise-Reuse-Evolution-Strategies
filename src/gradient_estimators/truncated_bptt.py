"""Truncated backprop through time (TBPTT)"""
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


@flax.struct.dataclass
class TruncatedBPTTAttributes(gradient_learner.GradientEstimatorState):
  current_unroll_states: TruncatedUnrollState


class TruncatedBPTT(gradient_learner.GradientEstimator):
  """Truncated Backpropagation through time"""

  def __init__(
      self,
      truncated_step: truncated_step.TruncatedStep,
      unroll_length: int,
      burn_in_length: int,
  ):
    """Initializer.

    Args:
      truncated_step: class containing functions for initializing and
        progressing an inner-training state.
      unroll_length: the truncation window size W.
      burn_in_length: how many steps to run unroll starting from the inner states
        returned from truncated_step.init_step_state to ensure all the states are
        obtained by using the current theta
    """
    self.truncated_step = truncated_step
    self.unroll_length = unroll_length
    self.burn_in_length = burn_in_length
    self._total_env_steps_used = 0


  def grad_est_name(self):
    return \
      ("TruncatedBPTT"
      f"_N={self.truncated_step.num_tasks},W={self.unroll_length}")

  def task_name(self):
    return self.truncated_step.task_name()

  @property
  def total_env_steps_used(self,):
    return self._total_env_steps_used

  @profile.wrap()
  def init_worker_state(self, worker_weights: gradient_learner.WorkerWeights,
                        key: PRNGKey) -> TruncatedBPTTAttributes:
    theta = worker_weights.theta
    outer_state = worker_weights.outer_state

    key1, key2 = jax.random.split(key, num=2)
    current_unroll_states = self.truncated_step.init_step_state(
        theta,
        outer_state,
        jax.random.split(key1, self.truncated_step.num_tasks),
        theta_is_vector=False)
    
    truncated_bptt_attributes = TruncatedBPTTAttributes(
      current_unroll_states=current_unroll_states,)
    # import ipdb; ipdb.set_trace()
    
    # a burn-in period to ensure the theta's have its states all unrolled by itself
    if self.burn_in_length > 0:
      rng = hk.PRNGSequence(key2)  # type: ignore
      for _ in range(self.burn_in_length):
        data = self.truncated_step.get_batch()
        truncated_bptt_attributes, _ = self.jitted_wrapped_unroll(
          theta=theta,
          state=truncated_bptt_attributes,
          key=next(rng),
          data=data,
          outer_state=outer_state,)
      
      # import ipdb; ipdb.set_trace()
      self._total_env_steps_used += int(jnp.sum(truncated_bptt_attributes.current_unroll_states.inner_step))

    # import ipdb; ipdb.set_trace()
    
    return truncated_bptt_attributes


  @profile.wrap()
  def compute_gradient_estimate(
      self,
      worker_weights,
      key: PRNGKey,
      state: TruncatedBPTTAttributes, # this is the same state returned by init_worker_state
      with_summary=False,
  ) -> Tuple[gradient_learner.GradientEstimatorOut, Mapping[str, jax.Array]]:

    data_sequence = self.truncated_step.get_batch(steps=self.unroll_length)
    mean_loss, g, next_state = self.truncated_gradient(
      worker_weights.theta,
      state,
      key,
      data_sequence,
      worker_weights.outer_state,
      )
    self._total_env_steps_used += int(self.truncated_step.num_tasks * self.unroll_length)

    output = gradient_learner.GradientEstimatorOut(
        mean_loss=mean_loss,
        grad=g,
        unroll_state=next_state,
        unroll_info=None)
    # import ipdb; ipdb.set_trace()
    return output, {}


  @functools.partial(
    jax.jit, static_argnums=(0,))
  def jitted_wrapped_unroll(
      self,
      theta: MetaParams, # a single theta
      state: TruncatedBPTTAttributes,
      key: chex.PRNGKey,
      data,
      outer_state,) -> Tuple[TruncatedBPTTAttributes, jax.Array]:
    current_unroll_states = state.current_unroll_states
    current_unroll_states, outs = self.truncated_step.unroll_step(
      theta=theta,
      unroll_state=current_unroll_states,
      key_list=jax.random.split(key, self.truncated_step.num_tasks),
      data=data,
      outer_state=outer_state,
      theta_is_vector=False)
    
    return \
      (
        TruncatedBPTTAttributes(
          current_unroll_states=current_unroll_states,
        ),
        outs.loss, # (num_particles,)
      )


  @functools.partial(
      jax.jit, static_argnums=(0,))
  def truncated_gradient(self,
      theta: MetaParams, # there is a single copy of theta
      state: TruncatedBPTTAttributes,
      key: PRNGKey,
      data_sequence: Any,
      outer_state: Any
    ):
    def L_truncated_single(theta):
      """
      Args:
          theta:
      returns:
          a list of losses obtained by unrolling the same theta for self.unroll_length times
      """
      init_state = state
      # defining the step function here so that we only pass in state and data
      # but not theta or outer_state
      def step(s, data):
        # s is (estimator_state, key)
        curr_state, base_key = s
        next_key, unroll_key = jax.random.split(base_key, 2)
        # data is a an element of data_sequence along the first dimension
        next_state, loss = self.jitted_wrapped_unroll(
          theta,
          curr_state,
          unroll_key,
          data,
          outer_state,)
        return (next_state, next_key), loss

      # note this end means with respect to the truncation window, not with respect
      # to the entire horizon length
      (end_state, end_key), losses = jax.lax.scan(step, (init_state, key), data_sequence)
      return end_state, losses

    def L_truncated_avg_single(theta):
      end_state, losses = L_truncated_single(theta)
      return jnp.mean(losses), (losses, end_state)

    value_and_grad_L_truncated_avg_single = \
      jax.value_and_grad(L_truncated_avg_single, has_aux=True)

    (avg_loss, (losses, end_state)), g = value_and_grad_L_truncated_avg_single(theta)
    # import ipdb; ipdb.set_trace()

    return avg_loss, g, end_state
