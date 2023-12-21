import abc
from typing import Any, Union, Sequence, Tuple

import jax
import jax.numpy as jnp
import flax.struct

InnerState = Any  # inner state
MetaParams = Any  # theta
Batch = Any
PRNGKey = Any
BoolArray = jax.Array

def key_vsplit(key_list: PRNGKey, num: int):
  # suppose num = N
  # here we assume key_list is a list of keys, jax.Array of shape (M, ...)
  # after doing vmap, we have shape (M, N, ...)
  # we then transpose the first two axes to get shape (N, M, ...)
  # return this transposed key_matrix
  return \
    jnp.transpose(jax.vmap(jax.random.split, in_axes=(0, None),)(key_list, num),  # type: ignore
                  axes=[1, 0] + list(key_list.shape[1:]))

@flax.struct.dataclass
class TruncatedUnrollOut:
  loss: jax.Array
  is_done: BoolArray
  task_param: Any
  iteration: jax.Array
  mask: jax.Array

class TruncatedStep(abc.ABC):
  """the new interface for truncated step"""
  @abc.abstractmethod
  def outer_init(self, key: PRNGKey) -> MetaParams:
    """return an outer init"""

  @abc.abstractmethod
  def task_name(self,) -> str:
    """return a string for the task name"""

  @property
  @abc.abstractmethod
  def num_tasks(self,) -> int:
    """return the number of tasks"""

  @property
  @abc.abstractmethod
  def T(self,) -> int:
    """Return the length of unrolled computation graph"""

  @abc.abstractmethod
  def get_batch(self, steps: Union[int, None] =None) -> Batch: 
    """if steps is None
        return a batch of data (first dimension being num_tasks)
    else
        return a sequence of batches of data (first dimension being steps,
                                              second dimension being num_tasks)"""

  @abc.abstractmethod
  def init_step_state(
    self,
    theta: MetaParams,
    outer_state: Any,
    key_list: PRNGKey,
    theta_is_vector: bool = False,
  ) -> InnerState:
    """"initialize the state for each particle
        total there are self.num_tasks particles
        here key_list is a list of keys (of number self.num_tasks) one for each particle"""

  @abc.abstractmethod
  def unroll_step(
    self,
    theta: MetaParams,
    unroll_state: InnerState,
    key_list: PRNGKey, # one key for each particle
    data, # a single step's data, first dimension being num_tasks
    outer_state,
    theta_is_vector: bool = False,
  ) -> Tuple[InnerState, TruncatedUnrollOut]:
    """unroll each particle by a single step, performs loss, and state reset (if necessary"""