USE_64 = False
from jax import config
config.update("jax_enable_x64", USE_64)

import functools
from typing import Any, Callable, Optional, Tuple, TypeVar, Union

import jax
import jax.numpy as jnp
import flax.struct
from src.task import dynamical_system as dynamical_system_lib

InnerState = Any  # inner state
MetaParams = Any  # theta
Batch = Any
PRNGKey = Any

@flax.struct.dataclass
class LorenzParameters:
  log_a: jax.Array  # single float jax array
  log_r: jax.Array  # single float jax array
  # log_b: float  # we don't learn this parameter

@flax.struct.dataclass
class LorenzState:
  # has to provide an inner_step field
  x: jax.Array
  y: jax.Array
  z: jax.Array
  _x_gt: jax.Array
  _y_gt: jax.Array
  _z_gt: jax.Array
  inner_step: jax.Array

class Lorenz_loga_logr_DynamicalSystem_FixedInit(dynamical_system_lib.DynamicalSystemDecomposable):
  """log_a, log_r parameterization of the lorenz system"""
  x_init = 1.2
  y_init = 1.3
  z_init = 1.6

  def __init__(
    self,
    dt: float,
    T: int,
    ground_truth_a: float, #pylint: disable=invalid-name
    ground_truth_r: float,
    ground_truth_b: float,
    init_a: float,
    init_r: float,):
    # init_b: float,) -> None:

    self.dt = dt
    self.T = T #pylint: disable=invalid-name
    # gt for ground_truth
    self.a_gt = ground_truth_a
    self.r_gt = ground_truth_r
    self.b_gt = ground_truth_b

    self.init_a = init_a
    self.init_r = init_r
    # self.init_b = init_b

    self.ground_truth_params = self.arb_to_theta(
        a=self.a_gt,
        r=self.r_gt,
        b=self.b_gt,
    )

  def name(self,):
    return ("Lorenz_loga_logr_FixedInit_"
            f"x={self.x_init}y={self.y_init}z={self.z_init}"
            f"dt={self.dt}T={self.T}_metainit_a={self.init_a}_r={self.init_r}")
  
  @staticmethod
  def meta_parameter_to_str(theta: MetaParams):
    return f"log(a): {theta.log_a}, log(r): {theta.log_r}"
  
  @staticmethod
  def meta_parameter_to_dict(theta: MetaParams):
    return {
      "log(a)": theta.log_a,
      "log(r)": theta.log_r,
    }

  def theta_to_arb(self, theta: MetaParams) -> tuple:
    # return jnp.exp(params.log_a), jnp.exp(params.log_r), jnp.exp(params.log_b)
    return jnp.exp(theta.log_a), jnp.exp(theta.log_r), self.b_gt
    # return 5.0 * jnp.exp(params.log_a), 23.0 * jnp.exp(params.log_r), 2.0 * jnp.exp(params.log_b)
    # return 5.0, jnp.exp(params.log_r), 2.0

  def arb_to_theta(self, a: float, r: float, b: float) -> MetaParams:
    del b # not used in the parameterization
    # return Parameters(log_r=jnp.log(r))
    # return Parameters(log_a=jnp.log(a), log_r=jnp.log(r), log_b=jnp.log(b))
    return LorenzParameters(
        log_a=jnp.log(a),
        log_r=jnp.log(r),
    )

  def ground_truth_theta(self,) -> MetaParams:
    return self.arb_to_theta(a=self.a_gt, b=self.b_gt, r=self.r_gt)
  
  def meta_init(self, key: PRNGKey) -> MetaParams:
    # wrong guessed values
    # we have to guess the correct value of b
    return self.arb_to_theta(a=self.init_a, r=self.init_r, b=self.b_gt)

  def inner_init(self, theta: MetaParams, key: PRNGKey) -> InnerState:
    # can change this function to allow for multiple inits
    x, y, z = self.x_init, self.y_init, self.z_init
    return \
        LorenzState(
          x=jnp.array([x]), # type: ignore
          y=jnp.array([y]), # type: ignore
          z=jnp.array([z]), # type: ignore
          _x_gt=jnp.array([x]), # type: ignore
          _y_gt=jnp.array([y]), # type: ignore
          _z_gt=jnp.array([z]), # type: ignore
          inner_step=jnp.array(0, dtype=jnp.int32)) # type: ignore
  
  @staticmethod
  def _inner_state_transition(x, y, z, a, r, b, dt):
    dx = a * (y - x) * dt
    dy = ((r - z) * x - y) * dt
    dz = (x * y - b * z) * dt

    return x + dx, y + dy, z + dz

  def unroll_without_reset(
            self, state: InnerState, theta: MetaParams, data,
            key: PRNGKey) -> InnerState:
    """perform unrolling without computing the loss"""
    a, r, b = self.theta_to_arb(theta)
    x, y, z = state.x, state.y, state.z
    x_gt, y_gt, z_gt = state._x_gt, state._y_gt, state._z_gt
    inner_step = state.inner_step

    # unroll the inner states
    x, y, z = self._inner_state_transition(x, y, z, a, r, b, self.dt)

    # unroll the ground truth state
    x_gt, y_gt, z_gt = \
      self._inner_state_transition(x_gt, y_gt, z_gt, self.a_gt, self.r_gt, self.b_gt, self.dt)
    
    return LorenzState(x=x, y=y, z=z, _x_gt=x_gt, _y_gt=y_gt, _z_gt=z_gt, inner_step=inner_step + 1)

  def loss_evaluation(self, state: InnerState, data, key: PRNGKey) -> jax.Array:
    loss = jnp.square(state.z[0] - state._z_gt[0])
    return loss

  def state_reset_if_necessary(self, state: InnerState, theta: MetaParams, key: PRNGKey) -> Tuple[bool, InnerState]:
    def true_fn():
      new_init_state = self.inner_init(theta, key)
      # the second element with respect to is_done
      return new_init_state

    def false_fn():
      # the second element with respect to is_done
      return state
    
    is_done = state.inner_step >= self.T
    next_state = jax.lax.cond(is_done, true_fn, false_fn,)
    return is_done, next_state

  @functools.partial(jax.jit, static_argnums=(0,))
  def _state_trajectory(self, theta: MetaParams, key: PRNGKey):
    # this function is no longer used right now
    # have to pass the second argument (even if it's None)
    def step(mega_state, _):
      # mega_state is a tuple of (inner_state, key)
      state, key = mega_state
      key, curr_key = jax.random.split(key, 2)
      new_state = self.unroll_without_reset(state, theta, None, curr_key)
      return (new_state, key), new_state

    init_key, unroll_key = jax.random.split(key, 2)
    init_state = self.inner_init(theta, init_key)
    # we have no input to feed into the step function but we specify the number of times
    # step() should be called
    _, state_trajectory = \
        jax.lax.scan(step, (init_state, unroll_key), None, length=self.T)
    state_trajectory = \
      jax.tree_util.tree_map(lambda a, b: jnp.concatenate((jnp.expand_dims(a, axis=0), b), axis=0),
                              init_state, state_trajectory)
    return state_trajectory
  

class Lorenz_loga_logr_DynamicalSystem_InfiniteGaussianInit(Lorenz_loga_logr_DynamicalSystem_FixedInit):
  def __init__(
    self,
    dt: float,
    T: int,
    ground_truth_a: float, #pylint: disable=invalid-name
    ground_truth_r: float,
    ground_truth_b: float,
    init_a: float,
    init_r: float,
    inner_init_sigma: float =0.0):

    super().__init__(
      dt=dt,
      T=T,
      ground_truth_a=ground_truth_a,
      ground_truth_r=ground_truth_r,
      ground_truth_b=ground_truth_b,
      init_a=init_a,
      init_r=init_r,
      )
    self.inner_init_sigma = inner_init_sigma

  def name(self,):
    return ("Lorenz_loga_logr_InfiniteGaussianInit_"
            f"x={self.x_init}y={self.y_init}z={self.z_init}_init_sigma={self.inner_init_sigma}_"
            f"dt={self.dt}T={self.T}_metainit_a={self.init_a}_r={self.init_r}")
  
  def inner_init(self, theta: MetaParams, key: PRNGKey) -> InnerState:
    # can change this function to allow for multiple inits
    init_perturbation = self.inner_init_sigma * jax.random.normal(key, shape=(3,))
    x = jnp.array([self.x_init]) + init_perturbation[0]
    y = jnp.array([self.y_init]) + init_perturbation[1]
    z = jnp.array([self.z_init]) + init_perturbation[2]
    return \
        LorenzState(x=x, # type: ignore
                    y=y, # type: ignore
                    z=z, # type: ignore
                    _x_gt=x, # type: ignore
                    _y_gt=y, # type: ignore
                    _z_gt=z, # type: ignore
                    inner_step=jnp.array(0, dtype=jnp.int32)) # type: ignore


class Lorenz_loga_logr_DynamicalSystem_FiniteGaussianInit(Lorenz_loga_logr_DynamicalSystem_FixedInit):
  def __init__(
    self,
    dt: float,
    T: int,
    ground_truth_a: float, #pylint: disable=invalid-name
    ground_truth_r: float,
    ground_truth_b: float,
    init_a: float,
    init_r: float,
    inner_init_sigma: float,
    n_init: int,
    key: PRNGKey):

    super().__init__(
      dt=dt,
      T=T,
      ground_truth_a=ground_truth_a,
      ground_truth_r=ground_truth_r,
      ground_truth_b=ground_truth_b,
      init_a=init_a,
      init_r=init_r,)
    self.inner_init_sigma = inner_init_sigma
    self.n_init = n_init
    # a matrix of shape n_init by 3 whose each row is a candidate perturbation
    # to be sampled
    self.init_perturbation_list = \
      self.inner_init_sigma * jax.random.normal(key, shape=(self.n_init, 3))

  def name(self,):
    return ("Lorenz_loga_logr_FiniteGaussianInit_"
            f"x={self.x_init}y={self.y_init}z={self.z_init}_"
            f"n-init={self.n_init}_init-sigma={self.inner_init_sigma}_"
            f"dt={self.dt}T={self.T}_metainit_a={self.init_a}_r={self.init_r}")
  
  def inner_init(self, theta: MetaParams, key: PRNGKey) -> InnerState:
    # () in the argument will make randint return an integer instead of list
    init_perturbation_idx = jax.random.randint(key, (), 0, self.n_init)
    init_perturbation = self.init_perturbation_list[init_perturbation_idx]
    x = jnp.array([self.x_init]) + init_perturbation[0]
    y = jnp.array([self.y_init]) + init_perturbation[1]
    z = jnp.array([self.z_init]) + init_perturbation[2]
    return \
        LorenzState(x=x, # type: ignore
                    y=y, # type: ignore
                    z=z, # type: ignore
                    _x_gt=x, # type: ignore
                    _y_gt=y, # type: ignore
                    _z_gt=z, # type: ignore
                    inner_step=jnp.array(0, dtype=jnp.int32)) # type: ignore



if __name__ == "__main__":
  """python3 -m src.task.lorenz"""
  import math
  import haiku

  dt = 0.005
  T = 2000
  lorenz_init_loga = 3.116
  lorenz_init_logr = 3.7
  ds = Lorenz_loga_logr_DynamicalSystem_FiniteGaussianInit(
    dt=dt,
    T=T,
    ground_truth_a=10.0, # the typically used value
    ground_truth_r=28.0, 
    ground_truth_b=8.0/3.0,
    init_a=math.exp(lorenz_init_loga),
    init_r=math.exp(lorenz_init_logr),
    inner_init_sigma=1.0,
    n_init=100,
    key=jax.random.PRNGKey(0),
  )
  theta = LorenzParameters(log_a=jnp.log(10.0), log_r=jnp.log(28.0))

  inner_state = ds.inner_init(theta, jax.random.PRNGKey(0))
  rng = haiku.PRNGSequence(0)
  outerloss_list = []

  unroll_function = jax.jit(ds.unroll)
  
  for i in range(T):
    print(i)
    out = unroll_function(inner_state, theta, None, next(rng))
    outerloss_list.append(out.loss)

  import ipdb; ipdb.set_trace()