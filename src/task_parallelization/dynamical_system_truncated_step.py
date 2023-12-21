"""multiple particle TruncatedStep for dynamical system"""
import functools
import numpy as np
import jax
import jax.numpy as jnp
from jaxlib import xla_client
import flax.struct

import src.task.dynamical_system as dynamical_system_lib
from src.task_parallelization import truncated_step
from src.task_parallelization.truncated_step import key_vsplit

from typing import Any, Callable, Optional, Tuple, TypeVar, Union

InnerState = Any  # inner state
MetaParams = Any  # theta
Batch = Any
PRNGKey = Any


class DynamicalSystemTruncatedStep(truncated_step.TruncatedStep):
  """
  TruncatedStep for initializing and unrolling multiples trajectories
  of dynamical system
  Note we don't keep track of environment steps here because unroll could be
  used in a scan/fori_loop which doesn't update the steps appropriately

  Here one assumes there is only an unroll function for dynamical_system
  """

  def __init__(
      self,
      dynamical_system: dynamical_system_lib.DynamicalSystem,
      num_tasks: int,  # number of tasks
      T,  # number of time steps,
      random_initial_iteration=False,
      truncation_window_size=1,
  ):
    assert isinstance(dynamical_system, dynamical_system_lib.DynamicalSystem), \
        f"{dynamical_system} is not of type DynamicalSystem"

    self.dynamical_system = dynamical_system
    self._num_tasks: int = num_tasks
    self._T = T
    self.random_initial_iteration = random_initial_iteration
    self.truncation_window_size = truncation_window_size

    self.init_fn_multi_theta = \
        jax.vmap(self.dynamical_system.inner_init, in_axes=(0, 0))
    self.init_fn_single_theta = \
        jax.vmap(self.dynamical_system.inner_init, in_axes=(None, 0))

    self.unroll_fn_multi_theta = \
        jax.vmap(self.dynamical_system.unroll, in_axes=(0, 0, 0, 0))
    self.unroll_fn_single_theta = \
        jax.vmap(self.dynamical_system.unroll, in_axes=(0, None, 0, 0))

    self.void_data_single_step = np.empty((self._num_tasks,))
    self.void_data_single_step.fill(np.nan)
    self.void_data_single_step = jnp.array(self.void_data_single_step)

  def outer_init(self, key) -> MetaParams:
    return self.dynamical_system.meta_init(key)

  def task_name(self):
    return self.dynamical_system.name()

  @property
  def num_tasks(self,) -> int:
    return self._num_tasks

  @property
  def T(self,) -> int:
    return self._T

  def get_batch(self, steps: Union[int, None] = None):
    # when step is None, still needs to unroll self.num_tasks number of
    # trajectories
    if steps is not None:
      if not (hasattr(self, "void_data")) or self.void_data.shape != (steps, self._num_tasks,):
        self.void_data = np.empty((steps, self._num_tasks,))
        self.void_data.fill(np.nan)
        self.void_data = jnp.array(self.void_data)
      # if not entering the if condition above, then there is saved void_data
      # object and we just return that
      return self.void_data
    else:
      return self.void_data_single_step

  def sample_random_initial_iteration(self, key_list):
    # return a list of random iterations (of shape (num_tasks,))
    # each iteration is a multiple of self.truncation_window_size
    def f(key, maxval):
      return jax.random.randint(key, (), 0, maxval, dtype=jnp.int32) * self.truncation_window_size
    if self.T % self.truncation_window_size == 0:
      max_val = self.T // self.truncation_window_size
    else:
      max_val = self.T // self.truncation_window_size + 1
    
    return  jax.vmap(f, in_axes=(0, None))(key_list, max_val)

  def init_step_state(
      self,
      theta: MetaParams,
      outer_state,
      key_list: PRNGKey,
      theta_is_vector=False) -> InnerState:
    init_key_list, iteration_key_list = key_vsplit(key_list, 2)
    if theta_is_vector:
      init_fn = self.init_fn_multi_theta
    else:
      init_fn = self.init_fn_single_theta
    
    inner_state = init_fn(theta, init_key_list)

    # whether to reset inner_step field of the inner_state
    # otherwise it should default to 0's for each trajectory
    # inner_step can be any integer between 0 and T - 1 (both inclusive)
    if self.random_initial_iteration:
      inner_step = self.sample_random_initial_iteration(iteration_key_list)
      inner_state = inner_state.replace(inner_step=inner_step)

    return inner_state

  def unroll_step(
      self,
      theta: MetaParams,
      unroll_state: InnerState,
      key_list: PRNGKey, # this should be a list of keys, one for each trajectory
      # assumed to have the leading axis corresponding to each trajectory (num_tasks total)
      data,
      outer_state,
      theta_is_vector: bool = False):

    inner_states = unroll_state

    if theta_is_vector:
      unroll_fn = self.unroll_fn_multi_theta
      # reset_fn = self.conditional_reset_multi_theta
    else:
      unroll_fn = self.unroll_fn_single_theta
      # reset_fn = self.conditional_reset_single_theta

    # move each particle forward by one step
    dynamical_system_out = unroll_fn(inner_states, theta, data, key_list)
    next_inner_states, losses, is_done = (
      dynamical_system_out.inner_state,
      dynamical_system_out.loss,
      dynamical_system_out.is_done
    )

    out = truncated_step.TruncatedUnrollOut(
        loss=losses,
        is_done=is_done, # type: ignore
        task_param=None,
        iteration=next_inner_states.inner_step,
        # currently we never return a meaningless loss value so the mask is all True
        mask=jnp.array([True for _ in range(self._num_tasks)]), # type: ignore
    )
    return next_inner_states, out


class DynamicalSystemDecomposableTruncatedStep(DynamicalSystemTruncatedStep):
  """
  TruncatedStep for initializing and unrolling multiples trajectories
  of dynamical system
  Note we don't keep track of environment steps here because unroll could be
  used in a scan/fori_loop which doesn't update the steps appropriately

  Here one assumes there is a three step function to make up unroll for dynamical_system
    unroll_without_reset
    loss_evaluation
    state_reset_if_necessary
  """

  def __init__(
      self,
      dynamical_system: dynamical_system_lib.DynamicalSystemDecomposable,
      num_tasks: int,  # number of tasks
      T,  # number of time steps,
      random_initial_iteration=False,
      truncation_window_size=1,
  ):

    assert isinstance(dynamical_system, dynamical_system_lib.DynamicalSystemDecomposable), \
        f"{dynamical_system} is not of type DynamicalSystemDecomposable"

    self.dynamical_system = dynamical_system
    self._num_tasks: int = num_tasks
    self._T = T
    self.random_initial_iteration = random_initial_iteration
    self.truncation_window_size = truncation_window_size

    self.init_fn_multi_theta = \
        jax.vmap(self.dynamical_system.inner_init, in_axes=(0, 0))
    self.init_fn_single_theta = \
        jax.vmap(self.dynamical_system.inner_init, in_axes=(None, 0))

    self.unroll_without_reset_fn_multi_theta = \
        jax.vmap(self.dynamical_system.unroll_without_reset, in_axes=(0, 0, 0, 0))
    self.unroll_without_reset_fn_single_theta = \
        jax.vmap(self.dynamical_system.unroll_without_reset, in_axes=(0, None, 0, 0))
    # we assume the loss evaluation doesn't depend on theta
    self.loss_evaluation_fn = \
        jax.vmap(self.dynamical_system.loss_evaluation, in_axes=(0, 0, 0))
    self.state_reset_if_necessary_fn_multi_theta = \
        jax.vmap(self.dynamical_system.state_reset_if_necessary, in_axes=(0, 0, 0),)
    self.state_reset_if_necessary_fn_single_theta = \
        jax.vmap(self.dynamical_system.state_reset_if_necessary, in_axes=(0, None, 0),)

    self.void_data_single_step = np.empty((self._num_tasks,))
    self.void_data_single_step.fill(np.nan)
    self.void_data_single_step = jnp.array(self.void_data_single_step)

  def init_step_state(self, theta: MetaParams, outer_state, key_list: PRNGKey, theta_is_vector=False):
    return super(DynamicalSystemDecomposableTruncatedStep, self).init_step_state(theta, outer_state, key_list, theta_is_vector)

  def unroll_without_reset_step(
        self,
        state: InnerState,
        theta: MetaParams,
        data,
        key_list: PRNGKey,
        theta_is_vector: bool) -> InnerState:
    if theta_is_vector:
      unroll_without_reset_fn = self.unroll_without_reset_fn_multi_theta
    else:
      unroll_without_reset_fn = self.unroll_without_reset_fn_single_theta
    
    return unroll_without_reset_fn(state, theta, data, key_list)

  def loss_evaluation_step(
        self,
        state: InnerState,
        data,
        key_list: PRNGKey) -> jax.Array:
    return self.loss_evaluation_fn(state, data, key_list)

  def state_reset_if_necessary_step(
        self,
        state: InnerState,
        theta: MetaParams,
        key_list: PRNGKey,
        theta_is_vector: bool) -> Tuple[jax.Array, InnerState]:
    if theta_is_vector:
      reset_if_necessary_fn = self.state_reset_if_necessary_fn_multi_theta
    else:
      reset_if_necessary_fn = self.state_reset_if_necessary_fn_single_theta
    return reset_if_necessary_fn(state, theta, key_list) # type: ignore

  def unroll_step(
      self,
      theta: MetaParams,
      unroll_state: InnerState,
      key_list: PRNGKey,
      # assumed to have the leading axis corresponding to each trajectory (num_tasks total)
      data,
      outer_state,
      theta_is_vector: bool = False) -> Tuple[InnerState, truncated_step.TruncatedUnrollOut]:
    
    unroll_without_reset_key_list, loss_key_list, reset_key_list = key_vsplit(key_list, 3)

    inner_states = unroll_state
    next_inner_states = self.unroll_without_reset_step(
      state=inner_states,
      theta=theta,
      data=data,
      key_list=unroll_without_reset_key_list,
      theta_is_vector=theta_is_vector)
    losses = self.loss_evaluation_step(
      state=next_inner_states,
      data=data,
      key_list=loss_key_list)
    is_done, next_inner_states = self.state_reset_if_necessary_step(
      state=next_inner_states,
      theta=theta,
      key_list=reset_key_list,
      theta_is_vector=theta_is_vector)
      
    out = truncated_step.TruncatedUnrollOut(
        loss=losses,
        is_done=is_done,
        task_param=None,
        iteration=next_inner_states.inner_step,
        # currently we never return a meaningless loss value so the mask is all True
        mask=jnp.array([True for _ in range(self._num_tasks)]), # type: ignore
    )
    return next_inner_states, out


class DynamicalSystemTruncatedStepMultiDevice(truncated_step.TruncatedStep):
  """
  TruncatedStep for initializing and unrolling multiples trajectories
  of dynamical system

  totally it simulates num_tasks, but distribute the computation over
  each device given in device_list
  """
  def __init__(
      self,
      dynamical_system: dynamical_system_lib.DynamicalSystem,
      num_tasks: int,  # number of tasks
      T,  # number of time steps,
      device_list: list[xla_client.Device], # list of devices from jax.devices()
      random_initial_iteration=False,
      truncation_window_size=1,
  ):
    assert isinstance(dynamical_system, dynamical_system_lib.DynamicalSystem), \
        f"{dynamical_system} is not of type DynamicalSystem"

    self.dynamical_system = dynamical_system
    self._num_tasks: int = num_tasks
    self._T = T
    self.random_initial_iteration = random_initial_iteration
    self.truncation_window_size = truncation_window_size
    self.device_list = device_list
    assert self._num_tasks % len(self.device_list) == 0, \
        f"num_tasks {self._num_tasks} must be divisible by number of devices {len(self.device_list)}"

    """
    init_fn_multi_theta would take in
    theta 
      a pytree with shape
      (n_devicees, n_tasks/n_devices,) + list(single_theta_shape)
      and
    init_key with shape (n_devices, n_tasks/n_devices,) + list(single_key_shape)
    return a pytree with shape (n_devices, n_tasks/n_devices,) + single state shape
    """
    self.init_fn_multi_theta = \
        jax.pmap(
          jax.vmap(self.dynamical_system.inner_init, in_axes=(0, 0)),
          in_axes=(0, 0), # type: ignore
          devices=self.device_list,)
    # init_fn_single_theta would take in a single theta with init_key of shape
    # described in init_fn_multi_theta
    self.init_fn_single_theta = \
        jax.pmap(
          jax.vmap(self.dynamical_system.inner_init, in_axes=(None, 0)),
          in_axes=(None,0), # type: ignore
          devices=self.device_list,)

    """
    unroll_fn_multi_theta would take in
    state 
      a pytree with shape (n_devices, n_tasks/n_devices,) + single state shape
    theta
      a pytree with shape
      (n_devicees, n_tasks/n_devices,) + list(single_theta_shape)
    data
      a pytree with shape
      (n_devicees, n_tasks/n_devices,) + list(single_data_shape)
    unroll_key with shape (n_devices, n_tasks/n_devices,) + list(single_key_shape)
    """
    self.unroll_fn_multi_theta = \
        jax.pmap(
          jax.vmap(self.dynamical_system.unroll, in_axes=(0, 0, 0, 0)),
          in_axes=(0, 0, 0, 0), # type: ignore
          devices=self.device_list,)
    # unroll_fn_single_theta is only different from unroll_fn_multi_theta in that
    # it takes in a single theta
    self.unroll_fn_single_theta = \
        jax.pmap(
          jax.vmap(self.dynamical_system.unroll, in_axes=(0, None, 0, 0)),
          in_axes=(0, None, 0, 0), # type: ignore
          devices=self.device_list,)

    self.void_data_single_step = np.empty((self._num_tasks,))
    self.void_data_single_step.fill(np.nan)
    self.void_data_single_step = jnp.array(self.void_data_single_step)
    
  def outer_init(self, key) -> MetaParams:
    return self.dynamical_system.meta_init(key)

  def task_name(self):
    return self.dynamical_system.name()

  @property
  def num_tasks(self,) -> int:
    return self._num_tasks

  @property
  def T(self,) -> int:
    return self._T

  def get_batch(self, steps: Union[int, None] = None):
    # gives a batch of data or a sequence of batch data
    # here the data is flat with respect to all the particles
    # when step is None, still needs to unroll self._num_tasks number of
    # trajectories
    if steps is not None:
      if not (hasattr(self, "void_data")) or self.void_data.shape != (steps, self._num_tasks,):
        self.void_data = np.empty((steps, self._num_tasks,))
        self.void_data.fill(np.nan)
        self.void_data = jnp.array(self.void_data)
      # if not entering the if condition above, then there is saved void_data
      # object and we just return that
      return self.void_data
    else:
      return self.void_data_single_step


  def reshape_for_pmap(self, pytree):
    # here we assume the first dimension of x is the particle dimension
    # and totally there should be num_tasks particles
    # after reshaping, we have the one more dimension for the devices
    # and product of the first two dimensions length is num_tasks
    return jax.tree_util.tree_map(
      lambda x: jnp.reshape(x, [len(self.device_list), -1,] + list(x.shape[1:])), pytree)
  
  def reshape_flatten(self, pytree):
    # here we assume the first dimension of x is the device dimension
    # and the first two dimensions' size's product is num_tasks
    # we merge the first two dimensions and return
    return jax.tree_util.tree_map(
      lambda x: jnp.reshape(x, [-1,] + list(x.shape[2:])),
      pytree)

  def sample_random_initial_iteration(self, key_list):
    # return a list of random iterations (of shape (num_tasks,))
    # each iteration is a multiple of self.truncation_window_size
    def f(key, maxval):
      return jax.random.randint(key, (), 0, maxval, dtype=jnp.int32) * self.truncation_window_size
    if self.T % self.truncation_window_size == 0:
      max_val = self.T // self.truncation_window_size
    else:
      max_val = self.T // self.truncation_window_size + 1
    
    return  jax.vmap(f, in_axes=(0, None))(key_list, max_val)

  def init_step_state(
      self,
      theta: MetaParams,
      outer_state,
      key_list: PRNGKey,
      theta_is_vector=False) -> InnerState:
    # here we assume 
    init_key_list, iteration_key_list = key_vsplit(key_list, 2)
    init_key_list = self.reshape_for_pmap(init_key_list)

    # import ipdb; ipdb.set_trace()
    if theta_is_vector:
      init_fn = self.init_fn_multi_theta
      theta = jax.tree_util.tree_map(
        self.reshape_for_pmap,
        theta)
    else:
      init_fn = self.init_fn_single_theta
    # inner_state of shape (n_devices, n_tasks/n_devices,) + single state shape
    inner_state = init_fn(theta, init_key_list)
    inner_state = self.reshape_flatten(inner_state)

    # whether to reset inner_step field of the inner_state
    # otherwise it should default to 0's for each trajectory
    # inner_step can be any integer between 0 and T - 1 (both inclusive)
    # import ipdb; ipdb.set_trace()
    if self.random_initial_iteration:
      inner_step = self.sample_random_initial_iteration(iteration_key_list)
      inner_state = inner_state.replace(inner_step=inner_step)

    # import ipdb; ipdb.set_trace()
    return inner_state

  def unroll_step(
      self,
      theta: MetaParams, # here theta is assumed either be a single theta
      # or have the first dimension being num of particles (num_tasks)
      unroll_state: InnerState,
      # this doesn't have an extra dimension for the devices
      # it just has the leading dimension for the particles (num_tasks)
      key_list: PRNGKey, # a single key
      # assumed to have the leading axis corresponding to each trajectory (num_tasks total)
      data, # assumed to have the leading axis corresponding to each particle (num_tasks total)
      outer_state,
      theta_is_vector: bool = False) -> Tuple[InnerState, truncated_step.TruncatedUnrollOut]:

    # import ipdb; ipdb.set_trace()
    inner_states = self.reshape_for_pmap(unroll_state)

    if theta_is_vector:
      unroll_fn = self.unroll_fn_multi_theta
      theta = self.reshape_for_pmap(theta)
      # reset_fn = self.conditional_reset_multi_theta
    else:
      unroll_fn = self.unroll_fn_single_theta
      # reset_fn = self.conditional_reset_single_theta

    unroll_key_list = self.reshape_for_pmap(key_list)

    data = self.reshape_for_pmap(data)

    # move each particle forward by one step
    dynamical_system_out = unroll_fn(inner_states, theta, data, unroll_key_list)
    next_inner_states, losses, is_done = (
      dynamical_system_out.inner_state,
      dynamical_system_out.loss,
      dynamical_system_out.is_done
    )
    next_inner_states = self.reshape_flatten(next_inner_states)
    losses = self.reshape_flatten(losses)
    is_done = self.reshape_flatten(is_done)
    inner_step = next_inner_states.inner_step
    
    # import ipdb; ipdb.set_trace()

    out = truncated_step.TruncatedUnrollOut(
        loss=losses,
        is_done=is_done,
        task_param=None,
        iteration=inner_step,
        # currently we never return a meaningless loss value so the mask is all True
        mask=jnp.array([True for _ in range(self._num_tasks)]), # type: ignore
    )
    return next_inner_states, out

class DynamicalSystemDecomposableTruncatedStepMultiDevice(DynamicalSystemTruncatedStepMultiDevice):
  def __init__(
      self,
      dynamical_system: dynamical_system_lib.DynamicalSystemDecomposable,
      num_tasks: int,  # number of tasks
      T: int,  # number of time steps,
      device_list: list[xla_client.Device], # list of devices from jax.devices()
      random_initial_iteration=False,
      truncation_window_size=1,
  ):
    super(DynamicalSystemDecomposableTruncatedStepMultiDevice, self).__init__(
      dynamical_system=dynamical_system,
      num_tasks=num_tasks,
      T=T,
      random_initial_iteration=random_initial_iteration,
      truncation_window_size=truncation_window_size,
      device_list=device_list,)

    assert isinstance(dynamical_system, dynamical_system_lib.DynamicalSystemDecomposable), \
        f"{dynamical_system} is not of type DynamicalSystemDecomposable"

    self.dynamical_system = dynamical_system
    self._num_tasks: int = num_tasks
    self._T = T
    self.random_initial_iteration = random_initial_iteration
    self.truncation_window_size = truncation_window_size
    self.device_list = device_list

    # init_fn_multi_theta would take in
    # theta 
    #   a pytree with shape
    #   (n_devicees, n_tasks/n_devices,) + list(single_theta_shape)
    #   and
    # init_key with shape (n_devices, n_tasks/n_devices,) + list(single_key_shape)
    # return a pytree with shape (n_devices, n_tasks/n_devices,) + single state shape
    self.init_fn_multi_theta = \
        jax.pmap(
          jax.vmap(self.dynamical_system.inner_init, in_axes=(0, 0)),
          in_axes=(0, 0),  # type: ignore
          devices=self.device_list,)
    # init_fn_single_theta would take in a single theta with init_key of shape
    # described in init_fn_multi_theta
    self.init_fn_single_theta = \
        jax.pmap(
          jax.vmap(self.dynamical_system.inner_init, in_axes=(None, 0)),
          in_axes=(None,0),  # type: ignore
          devices=self.device_list,)

    """
    unroll_without_reset_fn_multi_theta would take in
    state, a pytree of shape (n_devices, n_tasks/n_devices,) + single state shape
    theta, a pytree of shape (n_devices, n_tasks/n_devices,) + single theta shape
    data, a pytree of shape (n_devices, n_tasks/n_devices,) + single data shape
    key_list, a pytree of shape (n_devices, n_tasks/n_devices,) + single key shape
    returns
        next_state, a pytree of shape (n_devices, n_tasks/n_devices,) + single state shape
    """
    self.unroll_without_reset_fn_multi_theta = \
      jax.pmap(
        jax.vmap(self.dynamical_system.unroll_without_reset, in_axes=(0, 0, 0, 0)),
        in_axes=(0, 0, 0, 0),  # type: ignore
        devices=self.device_list,)
    self.unroll_without_reset_fn_single_theta = \
      jax.pmap(
        jax.vmap(self.dynamical_system.unroll_without_reset, in_axes=(0, None, 0, 0)),
        in_axes=(0, None, 0, 0),  # type: ignore
        devices=self.device_list,)
    """
    loss_evaluation_fn would take in
    state, a pytree of shape (n_devices, n_tasks/n_devices,) + single state shape
    data, a pytree of shape (n_devices, n_tasks/n_devices,) + single data shape
    key_list, a pytree of shape (n_devices, n_tasks/n_devices,) + single key shape
    return
       loss, a pytree of shape (n_devices, n_tasks/n_devices,) + single loss shape
    """
    self.loss_evaluation_fn = \
      jax.pmap(
        jax.vmap(self.dynamical_system.loss_evaluation, in_axes=(0, 0, 0)),
        in_axes=(0, 0, 0),  # type: ignore
        devices=self.device_list,)

    """
    state_reset_if_necessary_fn_multi_theta would take in
    state, a pytree of shape (n_devices, n_tasks/n_devices,) + single state shape
    theta, a pytree of shape (n_devices, n_tasks/n_devices,) + single theta shape
    key_list, a pytree of shape (n_devices, n_tasks/n_devices,) + single key shape
    return a tuple
      is_done, a pytree of shape (n_devices, n_tasks/n_devices,) + single is_done shape
      next_state, a pytree of shape (n_devices, n_tasks/n_devices,) + single next_state shape
    """
    self.state_reset_if_necessary_fn_multi_theta = \
      jax.pmap(
        jax.vmap(self.dynamical_system.state_reset_if_necessary, in_axes=(0, 0, 0)),
        in_axes=(0, 0, 0),  # type: ignore
        devices=self.device_list,)
    self.state_reset_if_necessary_fn_single_theta = \
      jax.pmap(
        jax.vmap(self.dynamical_system.state_reset_if_necessary, in_axes=(0, None, 0)),
        in_axes=(0, None, 0),  # type: ignore
        devices=self.device_list,)

  def init_step_state(self, theta: MetaParams, outer_state, key_list: PRNGKey, theta_is_vector=False) -> InnerState:
    return super(DynamicalSystemDecomposableTruncatedStepMultiDevice, self).init_step_state(theta, outer_state, key_list, theta_is_vector)

  def unroll_without_reset_step(
        self,
        state: InnerState, # here state is assumed to have the first dimension being num of particles (num_tasks)
        theta: MetaParams, # here theta is assumed either be a single theta, or have the first dimension being num of particles (num_tasks)
        data, # first dimension is num of particles (num_tasks)
        key_list: PRNGKey, # first dimension is num of particles (num_tasks)
        theta_is_vector: bool) -> InnerState:
    # import ipdb; ipdb.set_trace()
    inner_states = self.reshape_for_pmap(state)
    data = self.reshape_for_pmap(data)
    key_list = self.reshape_for_pmap(key_list)

    if theta_is_vector:
      unroll_without_reset_fn = self.unroll_without_reset_fn_multi_theta
      theta = self.reshape_for_pmap(theta)
    else:
      unroll_without_reset_fn = self.unroll_without_reset_fn_single_theta
    # next_inner_states have first dimension being num of devices
    next_inner_states = unroll_without_reset_fn(inner_states, theta, data, key_list)
    # now we merge the first two dimensions and return
    return self.reshape_flatten(next_inner_states)

  def loss_evaluation_step(
        self,
        state: InnerState, # here state is assumed to have the first dimension being num of particles (num_tasks)
        data, # first dimension is num of particles (num_tasks)
        key_list: PRNGKey, # first dimension is num of particles (num_tasks)
  ) -> jax.Array:
    # import ipdb; ipdb.set_trace()
    inner_states = self.reshape_for_pmap(state)
    data = self.reshape_for_pmap(data)
    key_list = self.reshape_for_pmap(key_list)
    # here losses is of shape (n_devices, n_tasks/n_devices,) + single loss shape
    losses = self.loss_evaluation_fn(inner_states, data, key_list)
    # we then flatten the first two dimensions into one and return
    return self.reshape_flatten(losses)

  def state_reset_if_necessary_step(
        self,
        state: InnerState, # here state is assumed to have the first dimension being num of particles (num_tasks)
        theta: MetaParams, # here theta is assumed either be a single theta, or have the first dimension being num of particles (num_tasks)
        key_list: PRNGKey, # first dimension is num of particles (num_tasks)
        theta_is_vector: bool) -> Tuple[jax.Array, InnerState]:
    # import ipdb; ipdb.set_trace()
    inner_states = self.reshape_for_pmap(state)
    key_list = self.reshape_for_pmap(key_list)

    if theta_is_vector:
      reset_if_necessary_fn = self.state_reset_if_necessary_fn_multi_theta
      theta = self.reshape_for_pmap(theta)
    else:
      reset_if_necessary_fn = self.state_reset_if_necessary_fn_single_theta
    # here is_done and next_inner_states will have first two dimensions being (n_devices, n_tasks/n_devices)
    is_done, next_inner_states = reset_if_necessary_fn(inner_states, theta, key_list)
    is_done = self.reshape_flatten(is_done)
    next_inner_states = self.reshape_flatten(next_inner_states)

    return is_done, next_inner_states

  def unroll_step(
      self,
      theta: MetaParams,
      unroll_state: InnerState, # have the first dimension being num of particles (num_tasks)
      key_list: PRNGKey, # have the first dimension being num of particles (num_tasks)
      # assumed to have the leading axis corresponding to each trajectory (num_tasks total)
      data, # have the first dimension being num of particles (num_tasks)
      outer_state,
      theta_is_vector: bool = False) -> Tuple[InnerState, truncated_step.TruncatedUnrollOut]:

    unroll_without_reset_key_list, loss_key_list, reset_key_list = key_vsplit(key_list, 3)

    inner_states = unroll_state
    next_inner_states = self.unroll_without_reset_step(
      state=inner_states,
      theta=theta,
      data=data,
      key_list=unroll_without_reset_key_list,
      theta_is_vector=theta_is_vector)
    losses = self.loss_evaluation_step(
      state=next_inner_states,
      data=data,
      key_list=loss_key_list)
    is_done, next_inner_states = self.state_reset_if_necessary_step(
      state=next_inner_states,
      theta=theta,
      key_list=reset_key_list,
      theta_is_vector=theta_is_vector)

    out = truncated_step.TruncatedUnrollOut(
        loss=losses,
        is_done=is_done,
        task_param=None,
        iteration=next_inner_states.inner_step,
        # currently we never return a meaningless loss value so the mask is all True
        mask=jnp.array([True for _ in range(self._num_tasks)]), # type: ignore
    )
    # import ipdb; ipdb.set_trace()
    return next_inner_states, out
