"""FullPESK with the same noise for every K steps, output a gradient estimate
after the entire episode finishes"""

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

from src.gradient_estimators import truncated_pesk


PRNGKey = jax.Array
MetaParams = Any
UnrollState = Any
TruncatedUnrollState = Any

import flax

class FullPESK(truncated_pesk.TruncatedPESK):
  """the offline version of GPESK, with new noise sampled every K unrolls."""

  def __init__(
      self,
      truncated_step: truncated_step.TruncatedStep,
      K: int,
      std: float,
      T: int,
      loss_normalize=False,
  ):
    """Initializer.

    Args:
      truncated_step: class containing functions for initializing and
        progressing a inner-training state.
      unroll_length: length of the unroll for each worker. 
      K: how frequent to generate a new epsilon
      std: standard deviation for ES.
      burn_in_length: how many steps to run unroll starting from the inner states
        returned from truncated_step.init_step_state to ensure all the states are
        obtained by using the current theta (+ epsilon)
      loss_normalize: whether to normalize the gradient by the std of the loss
    """
    super(FullPESK, self).__init__(
      truncated_step=truncated_step,
      unroll_length=T, # this argument is never supposed to be used because compute_gradient_estimate is overriden
      K=K,
      std=std,
      burn_in_length=0,
      loss_normalize=loss_normalize,
    )
    self.truncated_step = truncated_step
    self.K = K
    self.std = std
    self.T = T
    self.loss_normalize = loss_normalize
    self._total_env_steps_used = 0


  def grad_est_name(self):
    if self.K == self.T:
      name_list = [
      "FullES(PESK_impl)",
      f"N={self.truncated_step.num_tasks},K=T,W=T,sigma={self.std}"
    ]
      if self.loss_normalize:
        name_list.insert(1, "loss_normalized")
      return "_".join(name_list)
    else:
      name_list = [
        "FullPESK",
        f"N={self.truncated_step.num_tasks},K={self.K},W=T,sigma={self.std}"
      ]
      if self.loss_normalize:
        name_list.insert(1, "loss_normalized")
      return "_".join(name_list)
  
  @property
  def total_env_steps_used(self,):
    return self._total_env_steps_used

  @profile.wrap()
  def init_worker_state(self,
                        worker_weights: gradient_learner.WorkerWeights,
                        key: PRNGKey):
    # for FullPESK, we don't have any state to use across to compute_gradient_estimate calls
    return None

  @profile.wrap()
  def compute_gradient_estimate(
      self,
      worker_weights,
      key: PRNGKey,
      state, # this is the same state returned by init_worker_state so it should be None
      with_summary=False,
  ) -> Tuple[gradient_learner.GradientEstimatorOut, Mapping[str, jax.Array]]:

    data_sequence = self.truncated_step.get_batch(steps=self.T)
    mean_loss, g, loss_std = self.full_pesk(
      worker_weights,
      key=key,
      data_sequence=data_sequence,
      outer_state=worker_weights.outer_state,)
    self._total_env_steps_used += self.T * 2 * self.truncated_step.num_tasks

    output = gradient_learner.GradientEstimatorOut(
        mean_loss=mean_loss,
        grad=g,
        unroll_state=state, # this state should be None
        unroll_info=None)

    if with_summary:
      if loss_std is not None:
        return output, {"mean||loss_std": loss_std}
    return output, {}

  @functools.partial(
      jax.jit, static_argnums=(0,))
  def full_pesk(self,
      worker_weights, # there is a single copy of theta in the field .theta
      key: chex.PRNGKey,
      data_sequence: Any,
      outer_state: Any):

    T = tree_utils.first_dim(data_sequence)
    init_key, for_loop_key = jax.random.split(key)
    theta = worker_weights.theta

    # define the transition function for the for loop
    def body_fun(i, for_loop_state):
      # state is of type TruncatedPESKAttributes
      # pos_loss_cum is the loss sum over all time steps so far for each positive particle
      # g_cum is the gradient estimate sum over all particles are all time steps so far
      # count_cum is the number of gradient estimates summed in g_cum
      state, t, key, pos_loss_cum, neg_loss_cum, g_cum, count_cum = for_loop_state
      key, pesk_unroll_key = jax.random.split(key)
      data = jax.tree_util.tree_map(lambda x: x[i], data_sequence)
      (state, loss_sum_step, g_sum_step, count), (pos_loss, neg_loss) = \
        super(FullPESK, self).pesk_unroll(theta, state, pesk_unroll_key, data, outer_state)
      
      pos_loss_cum = pos_loss_cum + pos_loss
      neg_loss_cum = neg_loss_cum + neg_loss
      g_cum = jax.tree_util.tree_map(lambda x, y: x + y, g_cum, g_sum_step)
      count_cum = count_cum + count
      return state, t + 1, key, pos_loss_cum, neg_loss_cum, g_cum, count_cum
    
    # initialize the for_loop_state
    init_state = super(FullPESK, self).init_worker_state(worker_weights, init_key)  # type: ignore
    zero_theta = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), theta)
    for_loop_state = (
      init_state,
      jnp.array(0, dtype=jnp.int32),
      for_loop_key,
      jnp.zeros((self.truncated_step.num_tasks), dtype=jnp.float32),
      jnp.zeros((self.truncated_step.num_tasks), dtype=jnp.float32),
      zero_theta,
      jnp.array(0, dtype=jnp.int32),
    )

    (final_state, t, key, pos_loss_cum, neg_loss_cum, g_cum, count_cum) = \
      jax.lax.fori_loop(jnp.array(0, dtype=jnp.int32),
                        T,
                        body_fun,
                        for_loop_state)
    
    g = jax.tree_util.tree_map(lambda x: x / count_cum, g_cum)
    avg_L_pos = pos_loss_cum / t # shape (num_particles,)
    avg_L_neg = neg_loss_cum / t # shape (num_particles,)
    mean_loss = jnp.mean(avg_L_pos + avg_L_neg) / 2
    L_std = None

    if self.loss_normalize:
      L_std = jnp.std(jnp.concatenate([avg_L_pos, avg_L_neg], axis=0), axis=0)
      g = jax.tree_util.tree_map(lambda x: x * self.std / L_std, g)
    
    return mean_loss, g, L_std