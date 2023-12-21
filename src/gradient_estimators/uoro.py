"""The UORO gradient estimator, proposed in UNBIASED ONLINE RECURRENT OPTIMIZATION.
https://openreview.net/forum?id=rJQDjk-0b
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
from src.task_parallelization import dynamical_system_truncated_step
import chex


PRNGKey = jax.Array
MetaParams = Any
UnrollState = Any
TruncatedUnrollState = Any

import flax.struct
def shape(x):
  return jax.tree_util.tree_map(jnp.shape, x)

# here we are sampling in the dimension float_state_tilde
# which already has the first dimension as the number of particles
# input is pytree, a single key, std
@jax.jit
def sample_rademacher(variables: Any, key: chex.PRNGKey,) -> Any:
  flat, tree_def = jax.tree_util.tree_flatten(variables)
  rngs = jax.random.split(key, len(flat))
  perturbs = []
  for key, f in zip(rngs, flat):
    perturbs.append(jax.random.rademacher(key, shape=f.shape, dtype=f.dtype))
  return jax.tree_util.tree_unflatten(tree_def, perturbs)

def partition_state_into_float_nonfloat(state):
    partitions, unflattener = \
        tree_utils.partition([lambda k,v: jnp.asarray(v).dtype == jnp.float32 or jnp.asarray(v).dtype == jnp.float64], state, strict=False)
    return partitions, unflattener

def multiply_coefficient(X, C):
  # here we assume C is a vector of shape (num_tasks,)
  # while x is a vector of shape (num_tasks, float_state_dim)
  # we multiply each row of x with the corresponding element of c
  return jax.tree_util.tree_map(lambda x: 
                                            jnp.multiply(x,
                                                        jnp.reshape(C, list(C.shape) + [1] * (len(x.shape)- len(C.shape)))),
                                X)
def divide_coefficient(X, C):
  return jax.tree_util.tree_map(lambda x: 
                                            jnp.divide(x,
                                                      jnp.reshape(C, list(C.shape) + [1] * (len(x.shape)- len(C.shape)))),
                                X)

SMALL_NUMBER = 1e-7

@flax.struct.dataclass
class UOROAttributes(gradient_learner.GradientEstimatorState):
  unroll_states: TruncatedUnrollState
  theta_tilde: jax.Array # a pytree with leading dimension num_tasks
                          # rest of dimensions is the same as theta
  float_state_tilde: jax.Array # a pytree with leading dimension num_tasks
                          # rest of dimensions is the same as state

class UORO(gradient_learner.GradientEstimator):
  """
  The UORO gradient estimator.
  here the gradient estimator doesn't estimate the gradient of the initialized state
  with respect to theta (for example used in maml)
  we sample a new noise for every transition step
  algorithm see UORO paper.
  """

  truncated_step: dynamical_system_truncated_step.DynamicalSystemDecomposableTruncatedStepMultiDevice

  def __init__(
      self,
      truncated_step: dynamical_system_truncated_step.DynamicalSystemDecomposableTruncatedStepMultiDevice,
      unroll_length: int,
      burn_in_length: int,
  ):
    """Initializer.

    Args:
      truncated_step: class containing functions for initializing and
        progressing a inner-training state.
      unroll_length: length of the unroll truncation window W.
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
      ("UORO"
      f"_N={self.truncated_step.num_tasks},K=1,W={self.unroll_length}")

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
                        key: PRNGKey) -> UOROAttributes:
    key1, key2 = jax.random.split(key, num=2)
    tiled_thetas = self.theta_tile(worker_weights.theta)

    unroll_states = self.truncated_step.init_step_state(
            tiled_thetas,
            worker_weights.outer_state,
            jax.random.split(key1, self.truncated_step.num_tasks),
            theta_is_vector=True,)
    # unpack to know the shape of the float_unrollstates
    (float_unrollstates, nonfloat_unrollstates), unflattener = \
            partition_state_into_float_nonfloat(unroll_states)

    # leading dimension is num_tasks
    theta_tilde_zeros = jax.tree_util.tree_map(lambda x: jnp.zeros(shape=x.shape), tiled_thetas)
    # leading dimension is num_tasks
    float_state_tilde_zeros = jax.tree_util.tree_map(lambda x: jnp.zeros(shape=x.shape), float_unrollstates)

    uoro_attributes = UOROAttributes(
      unroll_states=unroll_states,
      theta_tilde=theta_tilde_zeros,
      float_state_tilde=float_state_tilde_zeros,)
    
    # a burn-in period to ensure the theta's have its states all unrolled by itself
    if self.burn_in_length > 0:
      rng = hk.PRNGSequence(key2)  # type: ignore
      for _ in range(self.burn_in_length):
        data = self.truncated_step.get_batch()
        curr_key = next(rng)
        (uoro_attributes, _, _, _) =\
          self.uoro_unroll(
              worker_weights.theta,
              uoro_attributes,
              curr_key,
              data,
              worker_weights.outer_state)
      self._total_env_steps_used += int(jnp.sum(uoro_attributes.unroll_states.inner_step))
    
    return uoro_attributes

  @profile.wrap()
  def compute_gradient_estimate(
      self,
      worker_weights,
      key: PRNGKey,
      state: UOROAttributes, # this is the same state returned by init_worker_state
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
      
      (state, loss_sum_step, g_sum_step, count) =\
        self.uoro_unroll(
            theta,
            state,
            curr_key,
            data,
            worker_weights.outer_state)
      
      # here there should be a debate over whether the multiple forward prop
      # and backprop should be counted as a single env step
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
  def uoro_unroll(
    self,
    theta: MetaParams, # there is a single copy of theta
    state: UOROAttributes, # this is the same state returned by init_worker_state
    key: chex.PRNGKey,
    data: Any, # single batch of data to be used for both pos and neg unroll
    outer_state: Any) -> Tuple[UOROAttributes, jax.Array, MetaParams, jax.Array]:
    # local function variable will be named *_state
    # global variable will be named *_unroll_states
    # assemble the complete state
    # here we use nonfloat_unrollstates defined outside this function

    tiled_thetas = self.theta_tile(theta) # repeated theta; one for each task
    rademacher_key, key = jax.random.split(key) # split the key for sampling rademacher noise
    key_list = jax.random.split(key, self.truncated_step.num_tasks) # split the key for each task
    unroll_without_reset_key_list, loss_evaluation_key_list, reset_key_list = \
      dynamical_system_truncated_step.key_vsplit(key_list, 3)

    # unpack the current state
    unroll_states, theta_tilde, float_state_tilde = \
      state.unroll_states, state.theta_tilde, state.float_state_tilde
    
    (float_unrollstates, nonfloat_unrollstates), unflattener = \
            partition_state_into_float_nonfloat(unroll_states)
    # sample a new noise perturbation in the state space for every unroll
    # this has the leading dimension of num_tasks
    # this noise should be a float
    state_noise_nu = sample_rademacher(float_unrollstates, rademacher_key,)

    def unroll_without_reset_state_theta_fn(float_state, thetas):
      # return both the next float state and the next unroll state

      # here we assemble the complete state (nonfloat part is outside this function)
      reassembled_unrollstate = tree_utils.partition_unflatten(
        unflattener=unflattener, part_values=[float_state, nonfloat_unrollstates])  # type: ignore
      next_state = self.truncated_step.unroll_without_reset_step(
          state=reassembled_unrollstate,
          theta=thetas,
          data=data,
          key_list=unroll_without_reset_key_list,
          theta_is_vector=True)
      (next_float_state, next_non_float_state), _ = partition_state_into_float_nonfloat(next_state)
      return (next_float_state, next_state)
    
    def unroll_without_reset_state_fn(float_state,):
      # return both the next float state and the next unroll state
      # here the only input variable is the float state
      # the theta is defined as tiled_theta outside this function
      return unroll_without_reset_state_theta_fn(float_state, tiled_thetas)

    def unroll_without_reset_theta_productfn(thetas):
      # only return next float state's inner product with state_noise_nu
      # float_state is given through float_unrollstates outside this function
      # return a single scalar
      next_float_state, _ = unroll_without_reset_state_theta_fn(float_unrollstates, thetas)
      loss = tree_utils.tree_inner_product(next_float_state, state_noise_nu)
      return loss

    ######### Step 1: state unrolling #########
    # jvp with_aux returns (primals_out, tangents_out, aux)
    # a1 of shape (num_tasks, float_state_dim)
    next_float_unroll_states, a1, next_unroll_states = \
      jax.jvp(unroll_without_reset_state_fn,
              primals=(float_unrollstates,),
              tangents=(float_state_tilde,),
              has_aux=True)
    # a2 of shape (num_tasks, float_state_dim)
    a2 = state_noise_nu

    # of shape (num_tasks, single_theta_dim)
    b1 = theta_tilde
    # of shape (num_tasks, single_theta_dim)
    b2 = jax.grad(unroll_without_reset_theta_productfn)(tiled_thetas)

    # compute the norms for renormalization
    a1_norm = jax.vmap(tree_utils.tree_norm, in_axes=(0,))(a1) # of shape (num_tasks,)
    a2_norm = jax.vmap(tree_utils.tree_norm, in_axes=(0,))(a2) # of shape (num_tasks,)
    b1_norm = jax.vmap(tree_utils.tree_norm, in_axes=(0,))(b1) # of shape (num_tasks,)
    b2_norm = jax.vmap(tree_utils.tree_norm, in_axes=(0,))(b2) # of shape (num_tasks,)

    rho_1 = jnp.sqrt(jnp.divide(b1_norm, a1_norm + SMALL_NUMBER)) + SMALL_NUMBER # of shape (num_tasks,)
    rho_2 = jnp.sqrt(jnp.divide(b2_norm, a2_norm + SMALL_NUMBER)) + SMALL_NUMBER # of shape (num_tasks,)

    # the norm of each row of new_a1 and new_b1, they should be the same
    # jax.vmap(tree_utils.tree_norm)(new_a1) and jax.vmap(tree_utils.tree_norm)(new_b1)
    new_a1 = multiply_coefficient(a1, rho_1)
    new_b1 = divide_coefficient(b1, rho_1)

    # check the norm of each row of new_a2 and new_b2, they should be the same
    # jax.vmap(tree_utils.tree_norm)(new_a2) and jax.vmap(tree_utils.tree_norm)(new_b2)
    new_a2 = multiply_coefficient(a2, rho_2)
    new_b2 = divide_coefficient(b2, rho_2)

    next_float_state_tilde = jax.tree_util.tree_map(lambda x, y: x + y, new_a1, new_a2)
    next_theta_tilde = jax.tree_util.tree_map(lambda x, y: x + y, new_b1, new_b2)

    def loss_evaluation(float_state):
      # we reassemble the next_unroll_states inside
      (_, nonfloat_state), unflattener = \
            partition_state_into_float_nonfloat(next_unroll_states)
      reassembled_unrollstate = tree_utils.partition_unflatten(
        unflattener=unflattener, part_values=[float_state, nonfloat_state])  # type: ignore
      # here the loss should be of shape (num_tasks,)
      loss = self.truncated_step.loss_evaluation_step(
        state=reassembled_unrollstate,
        data=data,
        key_list=loss_evaluation_key_list,
      )
      # here each state only contributes to its own loss
      return jnp.sum(loss)

    ######### Step 2: loss computation #########
    # If has_aux is True then jax.grad returns a pair of (gradient, auxiliary_data).
    loss_sum_step, dl_dfloatstate = jax.value_and_grad(loss_evaluation)(next_float_unroll_states)
    # dl_dfloatstate's shape is the same as next_float_unroll_states
    # below: of shape (num_tasks,)
    dl_dstate_inner_product_w_next_state_tilde = \
      jax.vmap(tree_utils.tree_inner_product, in_axes=(0, 0,))(dl_dfloatstate, next_float_state_tilde)
    # of shape (num_tasks, theta_dim)
    g = multiply_coefficient(next_theta_tilde, dl_dstate_inner_product_w_next_state_tilde)

    ######### Step 3: state resetting #########
    is_done, next_unroll_states = self.truncated_step.state_reset_if_necessary_step(
      state=next_unroll_states,
      theta=tiled_thetas,
      key_list=reset_key_list,
      theta_is_vector=True,)
    # sum over all particles (each particle is working on one trajectory)
    g_sum_step = jax.tree_util.tree_map(lambda x: jnp.sum(x, axis=0), g)
    # count keeps track of how many gradient estimates are summed in this unroll
    count = jnp.array(self.truncated_step.num_tasks)

    # for the particle that has reached its end, we clear its theta_tilde and state_tilde
    def clear_row(x,):
      # set eps to 0 if is_done is True
      reshape_isdone = jnp.reshape(is_done,
                                    [self.truncated_step.num_tasks] + [1] * (len(x.shape)-1))
      return x * (1 - reshape_isdone)
    next_theta_tilde = jax.tree_util.tree_map(clear_row, next_theta_tilde)
    next_float_state_tilde = jax.tree_util.tree_map(clear_row, next_float_state_tilde)

    return (
      UOROAttributes(
              unroll_states=next_unroll_states,
              theta_tilde=next_theta_tilde,
              float_state_tilde=next_float_state_tilde,),
      loss_sum_step,
      g_sum_step,
      count)


class FullUORO(UORO):
  def __init__(
    self,
    truncated_step: dynamical_system_truncated_step.DynamicalSystemDecomposableTruncatedStepMultiDevice,
    T: int,
  ):
    super(FullUORO, self).__init__(
      truncated_step=truncated_step,
      unroll_length=T,
      burn_in_length=0,
    )

  def grad_est_name(self):
    # there is no sigma parameter for TruncatedForwardMode
    return \
      ("FullUORO"
      f"_N={self.truncated_step.num_tasks},K=1,W=T")