"""BPTT (FullGradient)."""
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

import flax

# this is different from vector_sample_perturbations
# as we don't need the positive and negatively perturbed thetas
# (theta, key, std) # only keys are different over multiple samples
sample_multiple_perturbations = jax.jit(jax.vmap(common.sample_perturbations, in_axes=(None, 0, None)))


class FullGradient(gradient_learner.GradientEstimator):
  """ReverseMode differentiation through the entire episode
  a.k.a Backpropagation through time (BPTT)"""

  def __init__(
      self,
      truncated_step: truncated_step.TruncatedStep,
      T: int,
  ):
    """Initializer.

    Args:
      truncated_step: class containing functions for initializing and
        progressing a inner-training state.
      T: the horizon length
    """
    self.truncated_step = truncated_step
    self.T = T
    self._total_env_steps_used = 0

  def grad_est_name(self):
    return \
      ("FullGradient"
      f"_N={self.truncated_step.num_tasks},W=T")
  
  def task_name(self):
    return self.truncated_step.task_name()

  @property
  def total_env_steps_used(self,):
    return self._total_env_steps_used

  @profile.wrap()
  def init_worker_state(self, worker_weights: gradient_learner.WorkerWeights,
                        key: PRNGKey):
    
    return None

  @profile.wrap()
  def compute_gradient_estimate(
      self,
      worker_weights,
      key: PRNGKey,
      state, # this is the same state returned by init_worker_state
      with_summary=False,
  ) -> Tuple[gradient_learner.GradientEstimatorOut, Mapping[str, jax.Array]]:

    data_sequence = self.truncated_step.get_batch(steps=self.T)
    mean_loss, g = self.full_gradient(
      theta=worker_weights.theta,
      key=key,
      data_sequence=data_sequence,
      outer_state=worker_weights.outer_state,
      )
    self._total_env_steps_used += self.T * self.truncated_step.num_tasks

    output = gradient_learner.GradientEstimatorOut(
        mean_loss=mean_loss,
        grad=g,
        unroll_state=state,
        unroll_info=None)

    return output, {}

  @functools.partial(
      jax.jit, static_argnums=(0,))
  def full_gradient(self,
      theta: MetaParams, # there is a single copy of theta
      key: chex.PRNGKey,
      data_sequence: Any,
      outer_state: Any
    ):

    def L_all_single(theta):
      """
      Args:
          thetas:
      returns:
          a matrix of losses obtained by unrolling the same theta for each time step
            (T, num_particles)
      """
      init_key, unroll_key = jax.random.split(key, 2)
      init_unroll_state = self.truncated_step.init_step_state(
              theta,
              outer_state,
              jax.random.split(init_key, self.truncated_step.num_tasks),
              theta_is_vector=False)
      
      def step(scan_state, data):
          inner_unroll_state, t, key = scan_state
          key1, key2 = jax.random.split(key, num=2)
          
          new_inner_unroll_state, inner_unroll_out = self.truncated_step.unroll_step(
                  theta=theta, # pick the theta at the time step t.
                  unroll_state=inner_unroll_state,
                  key_list=jax.random.split(key1, self.truncated_step.num_tasks),
                  data=data,
                  outer_state=outer_state,
                  theta_is_vector=False)
          new_scan_state = (new_inner_unroll_state, t+1, key2)
          return new_scan_state, inner_unroll_out.loss
          
      _, losses = jax.lax.scan(step, (init_unroll_state, 0, unroll_key), data_sequence)
      return losses

    def L_avg_single(theta):
      # average losses over all time steps and all particles
      return jnp.mean(L_all_single(theta))
    
    value_and_grad_L_avg_single = jax.value_and_grad(L_avg_single)

    avg_loss, g = value_and_grad_L_avg_single(theta)

    return avg_loss, g
