import abc
import functools
import jax
import jax.numpy as jnp
import flax.struct

from typing import Any, Callable, Optional, Tuple, TypeVar, Union

InnerState = Any  # inner state
MetaParams = Any  # theta
Batch = Any  # not needed as we assume DynamicalSystem keeps track of its own data
PRNGKey = Any
Bool = jax.Array  # a single boolean (before vmap)

@flax.struct.dataclass
class DynamicalSystemUnrollOut:
  inner_state: InnerState # inner_state object needs to have a field inner_step
  loss: jax.Array
  is_done: Bool # without vmap, this is a jnp.array of a single True or False


class DynamicalSystem(abc.ABC):
  """
  the dynamical system abc that focuses on a **single** trajectory
  """
  @abc.abstractmethod
  def name(self,) -> str:
    """the name of the dynamical system"""

  @abc.abstractmethod
  def meta_init(self, key: PRNGKey) -> MetaParams:
    """
    initialize a theta for meta-optimization
    call this before calling self.inner_init
    """
  
  def init(self, key: PRNGKey) -> MetaParams:
    """the function to interface with GradientLearner"""
    # not an abstract method
    return self.meta_init(key)

  @abc.abstractmethod
  def inner_init(self, theta: MetaParams, key: PRNGKey) -> InnerState:
    """
    initialize a single trajectory's inner state at time step 0
    taking theta as an argument considers the cases where theta is used for init

    InnerState has to contain a field inner_step
      because:
        1. the loss evaluation needs to know at what time step the state is reached
        2. DynamicalSystemTruncatedStep could reset the inner_step field of
          InnerState to ensure particle unlocked steps.
    """

  @abc.abstractmethod
  def unroll(
        self, state: InnerState, theta: MetaParams, data,
        key: PRNGKey) -> DynamicalSystemUnrollOut:
    """
    run the dynamical system forward by 1 step
    if not reached the last time step of the horizon:
      return the next inner state, the loss incurred, False (is_done)
    else:
      return the new initialized state, the loss incurred, True (is_done)
        (from the last state from the previous horizon)
    """
  
  @staticmethod
  def meta_parameter_to_str(theta: MetaParams) -> str:
    """return a string representation of the meta-parameter"""
    return str(theta)
  
  @staticmethod
  def meta_parameter_to_dict(theta: MetaParams) -> dict[str, Any]:
    """return a dict mapping string variable name to the value"""
    return theta


class DynamicalSystemDecomposable(DynamicalSystem):
  """
  the dynamical interface that assumes the original unroll function can be
  decomposed into 3 steps:
    1. unroll_without_reset -> InnerState
    2. loss_evaluation -> jax.Array # a single scalar give a single state
    3. state_reset_if_necessary -> InnerState # reset the state if at the end of horizon
  """
  @abc.abstractmethod
  def unroll_without_reset(
          self, state: InnerState, theta: MetaParams, data,
          key: PRNGKey) -> InnerState:
    """
    run the dynamical system forward by 1 step
    never resets the inner state
    """

  @abc.abstractmethod
  def loss_evaluation(
    self, state: InnerState, data, key: PRNGKey,) -> jax.Array:
    """
    compute a scalar loss given the single inner state
    """

  @abc.abstractmethod
  def state_reset_if_necessary(
    self, state: InnerState, theta: MetaParams, key: PRNGKey,) -> Tuple[Bool, InnerState]:
    """
    reset the inner state if at the end of horizon for this particle
    """

  def unroll(
    self, state: InnerState, theta: MetaParams, data, key: PRNGKey) -> DynamicalSystemUnrollOut:
    """
    run the dynamical system forward by 1 step, compute the outer loss, and reset
    if necessary
    """
    unroll_without_reset_key, loss_evaluation_key, reset_key = jax.random.split(key, 3)
    next_state = self.unroll_without_reset(state, theta, data, unroll_without_reset_key)
    loss = self.loss_evaluation(next_state, data, loss_evaluation_key)
    is_done, next_state = self.state_reset_if_necessary(next_state, theta, reset_key)

    return DynamicalSystemUnrollOut(
      inner_state=next_state,
      loss=loss,
      is_done=is_done,
    )


