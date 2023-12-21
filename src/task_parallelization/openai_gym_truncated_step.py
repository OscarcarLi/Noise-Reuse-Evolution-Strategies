"""multiple particle TruncatedStep for cpu-based OpenAI Gym environments"""
import functools
import numpy as np
import jax
import jax.numpy as jnp
import flax.struct
import haiku
import time

import gymnasium
import gymnasium.vector
from src.task_parallelization import truncated_step
from src.task_parallelization.truncated_step import key_vsplit
from src.utils.mujoco_env_utils import (
  MUJOCO_ENV_NAMES, EnvCreator, copy_mujoco_sync_venv)
from collections import namedtuple

from typing import Any, Callable, Optional, Tuple, TypeVar, Union

InnerState = Any  # inner state
MetaParams = Any  # theta
Batch = Any
PRNGKey = Any
OuterState = Any

OuterBatch = Any
InnerBatch = Any

OpenAIGymTruncatedInnerState = \
  namedtuple('OpenAITruncatedInnerState', ['env', 'inner_state', 'inner_step', 'info'])


@functools.partial(jax.jit, static_argnums=(0,))
def policy_vec_apply_vec_theta(policy, params, key_list, state,):
  return jax.vmap(policy.apply, in_axes=(0, 0, 0))(params, key_list, state)

@functools.partial(jax.jit, static_argnums=(0,))
def policy_vec_apply_single_theta(policy, params, key_list, state,):
  return jax.vmap(policy.apply, in_axes=(None, 0, 0))(params, key_list, state)


class OpenAIGymTruncatedStep(truncated_step.TruncatedStep):
  def __init__(
    self,
    env_name: str,
    num_tasks: int,
    T: int, # here there is no way to enforce this T is actually the same as the
    # the env's T because gy.make doesn't accept a T argument
    policy: haiku.Transformed,
    random_initial_iteration: bool=False,
    truncation_window_size: int=1,
    init_theta=None,
    immutable_state: bool=True,):

    self.env_name = env_name
    env = gymnasium.make(self.env_name)
    self.observation_dim = env.observation_space.shape[0]  # type: ignore
    self.action_dim = env.action_space.shape[0]  # type: ignore
    
    self._num_tasks = num_tasks
    self._T = T
    self.policy = policy # haiku function tuple
    self.random_initial_iteration = random_initial_iteration
    self.truncation_window_size = truncation_window_size

    self.init_theta = init_theta

    self.void_data_single_step = np.empty((self._num_tasks,))
    self.void_data_single_step.fill(np.nan)
    self.void_data_single_step = jnp.array(self.void_data_single_step)

    self.immutable_state = immutable_state

  def init(self, key: PRNGKey) -> MetaParams:
    # this is to be used with gradient_learner.SingleMachineGradientLearner
    if self.init_theta is None:
      theta_0 = self.outer_init(key)
    else:
      theta_0 = self.init_theta
    print(theta_0)
    return theta_0

  def outer_init(self, key: PRNGKey) -> MetaParams:
    return self.policy.init(key, jnp.zeros(self.observation_dim))

  def task_name(self):
    return self.env_name

  def name(self,):
    # this is also for cases when this truncated step is treated as a task
    return self.env_name
  
  @property
  def num_tasks(self,) -> int:
    return self._num_tasks

  @property
  def T(self,) -> int:
    return self._T

  def get_batch(self, steps: Union[int, None] = None) -> Batch:
    # when step is None, still needs to unroll self.num_tasks number of
    # trajectories
    if steps is not None:
      if not (hasattr(self, "void_data")) or self.void_data.shape != (steps, self.num_tasks,):
        self.void_data = np.empty((steps, self.num_tasks,))
        self.void_data.fill(np.nan)
        self.void_data = jnp.array(self.void_data)
      # if not entering the if condition above, then there is saved void_data
      # object and we just return that
      return self.void_data
    else:
      # this is guaranteed to be created upon truncated step initialization
      return self.void_data_single_step

  def sample_random_initial_iteration(self, key_list):
    # return a list of random iterations (of shape (num_tasks,))
    # each iteration is a multiple of self.truncation_window_size
    def f(key, maxval):
      # the sampling doesn't include maxval
      return \
        jax.random.randint(key, (), 0, maxval,
                           dtype=jnp.int32) * self.truncation_window_size

    if self.T % self.truncation_window_size == 0:
      max_val = self.T // self.truncation_window_size
    else:
      max_val = self.T // self.truncation_window_size + 1
    
    return  jax.vmap(f, in_axes=(0, None))(key_list, max_val)
  
  def init_step_state(self, 
                      theta: MetaParams,
                      outer_state: OuterState,
                      key_list: PRNGKey,
                      theta_is_vector=False,) -> InnerState:

    # init key for resetting each environment
    # iteration_key is for choosing the initial iteration for each particle
    # sample_action_key is for sampling actions
      # because we need to sample actions for each particle of different number
      # of times, we don't allocate a seperate key for each particle
    init_key_list, iteration_key_list, sample_action_key_list = key_vsplit(key_list, 3)
    
    # whether to reset inner_step field of the inner_state
    # otherwise it should default to 0's for each trajectory
    # inner_step can be any integer between 0 and T - 1 (both inclusive)
    if self.random_initial_iteration:
      inner_step = self.sample_random_initial_iteration(iteration_key_list)
      inner_step = [int(i) for i in inner_step]
    else:
      inner_step = [0 for _ in range(self.num_tasks)]

    # a list of environments
      # here we cannot put env_vec as a field of self because we will use it multiple times
      # at least once for positive particles and once for negative particles
    env_vec = []
    obs_vec = [] # a list of observations
    info_vec = [] # a list of info
    for i in range(self.num_tasks):
      if theta_is_vector:
        theta_i = jax.tree_util.tree_map(lambda x: x[i], theta)
      else:
        theta_i = theta
      sample_action_key = sample_action_key_list[i]

      env = gymnasium.make(self.env_name)
      # first create an integer seed for numpy
      np_seed = int(jax.random.randint(init_key_list[i], (), 0, 1e6))
      obs, info = env.reset(seed=np_seed)

      for j in range(inner_step[i]):
        sample_action_key, action_key = jax.random.split(sample_action_key)
        action = np.array(self.policy.apply(theta_i, action_key, obs))
        # here because inner_step is smaller than T, it shouldn't have reached the end of the episode
        obs, reward, terminated, truncated, info = env.step(action)
        assert not terminated, "inner_step is smaller than T, so it shouldn't have reached the end of the episode"
        # truncated can be used to end the episode prematurely before a terminal state is reached.
        # TODO: if we are using envs with unhealthy condition, we should handle this better
        assert not truncated, "inner_step is smaller than T, so it shouldn't have reached the end of the episode"
        # print(env._elapsed_steps)
      
      env_vec.append(env)
      obs_vec.append(obs)
      info_vec.append(info)

    # create a vector of environments generating functions () -> env
    # we CANNOT simply use lambda expressions! because the evaluation of lambda expressions
    # happen at the time of calling, and thus the environments will be all the same
    # (the last one)
    env_fns = [EnvCreator(env=env) for env in env_vec]

    # currently we are doing a customized implementation of vector environments
    # deepcopy if we require immutable state
    # AsyncVectorEnv is difficult to deepcopy (because cannot be pickled and 
    # also don't have .envs attribute)
    if self.immutable_state:
      venv = gymnasium.vector.SyncVectorEnv(env_fns)
    else:
      # Async cann't be deepcopied
      # venv = gymnasium.vector.AsyncVectorEnv(env_fns)
      venv = gymnasium.vector.SyncVectorEnv(env_fns)
    # NOTE: when we are using vector envs, we don't need to self reset the envs
    # if would happen automatically when truncated or terminated is True

    ## for checking the vector of envs creation's correctness
    # for env in venv.envs:
    #   print(env._elapsed_steps)
    return OpenAIGymTruncatedInnerState(env=venv, inner_state=np.array(obs_vec), inner_step=jnp.array(inner_step), info=info_vec)

  def unroll_step(
    self,
    theta: MetaParams,
    unroll_state: InnerState,
    key_list: PRNGKey,
    data: InnerBatch,
    outer_state: OuterState,
    theta_is_vector: bool = False) -> Tuple[InnerState, truncated_step.TruncatedUnrollOut]:

    # reset key for the environments that have reached the end of its current episode
    # reset_key, action_key = jax.random.split(key)
    # reset_key_list = jax.random.split(reset_key, self.num_tasks)
    action_key_list = key_list

    venv, vobs, inner_step = \
      unroll_state.env, unroll_state.inner_state, unroll_state.inner_step
    # import ipdb; ipdb.set_trace()
    
    # start_copy_time = time.time()
    if self.immutable_state:
      # if there exists an unroll_state.inner_step that is multiple of self.truncation_window_size do the copying
      # print("inner_step", inner_step)
      if jnp.any(jnp.equal(jnp.mod(inner_step, self.truncation_window_size), 0)):
        assert isinstance(venv, gymnasium.vector.SyncVectorEnv), "currently only support sync vector envs, not AsyncVectorEnv"
        assert self.env_name in MUJOCO_ENV_NAMES, "currently only support mujoco envs"
        venv = copy_mujoco_sync_venv(venv)
    # end_copy_time = time.time()
    # print("copy_time", end_copy_time - start_copy_time)
    
    if theta_is_vector:
      policy_apply_fn = policy_vec_apply_vec_theta
    else:
      policy_apply_fn = policy_vec_apply_single_theta
    
    vaction = policy_apply_fn(self.policy, theta, action_key_list, vobs,)
    vobs, vreward, vterminated, vtruncated, vinfo = venv.step(np.array(vaction))
    vdone = np.logical_or(vterminated, vtruncated)

    inner_step = [inner_step[i] + 1 if not vdone[i] else jnp.array(0, dtype=jnp.int32) for i in range(self.num_tasks)]

    loss_vec = jnp.array(-vreward)
    done_vec = jnp.array(vdone)
    
    # we don't update venv as the environment objects are updated automatically
    next_unroll_states = \
      OpenAIGymTruncatedInnerState(
        env=venv, inner_state=vobs, inner_step=jnp.array(inner_step), info=vinfo)

    out = truncated_step.TruncatedUnrollOut(
        loss=loss_vec,
        is_done=done_vec,
        task_param=None,
        iteration=jnp.array(inner_step),
        # currently we never return a meaningless loss value so the mask is all True
        mask=jnp.array([True for _ in range(self.num_tasks)]), # type: ignore
    )
    end_unroll_time = time.time()
    # print("unroll_time", end_unroll_time - end_copy_time)
    
    return next_unroll_states, out

if __name__ == "__main__":
  from jax import config
  config.update('jax_platform_name', 'cpu')

  env_name = "Swimmer-v4"
  T = 1000
  env = gymnasium.make(env_name)
  policy = haiku.transform(
    lambda x: haiku.Linear(env.action_space.shape[0], with_bias=False, w_init=haiku.initializers.Constant(0.0))(x))  # type: ignore
  theta = policy.init(jax.random.PRNGKey(0), jnp.zeros(env.observation_space.shape))
  print(theta)

  # truncated_step = OpenAIGymTruncatedStep(
  #   env_name=env_name,
  #   num_tasks=10,
  #   T=T,
  #   policy=policy,
  #   random_initial_iteration=False,
  #   truncation_window_size=1,)
  

  
  # unroll_states = truncated_step.init_step_state(
  #     theta=theta,
  #     outer_state=None,
  #     key=jax.random.PRNGKey(0),
  #     theta_is_vector=False,)
  # # import ipdb; ipdb.set_trace()
  
  # for i in range(T-1):
  #   unroll_states, out = truncated_step.unroll_step(theta, unroll_states, jax.random.PRNGKey(0), None, None, theta_is_vector=False)
  
  # # import ipdb; ipdb.set_trace()
  # unroll_states, out = truncated_step.unroll_step(theta, unroll_states, jax.random.PRNGKey(0), None, None, theta_is_vector=False)
  # # import ipdb; ipdb.set_trace()

  W = 100
  N = 40
  t_step = OpenAIGymTruncatedStep(
    env_name=env_name,
    num_tasks=N,
    T=T,
    policy=policy,
    random_initial_iteration=True,
    truncation_window_size=W,)
  unroll_states = t_step.init_step_state(
    theta=theta,
    outer_state=None,
    key_list=jax.random.split(jax.random.PRNGKey(0), N),
    theta_is_vector=False,)
  import ipdb; ipdb.set_trace()
  for i in range(T//W):
    unroll_states, out = t_step.unroll_step(theta, unroll_states,
                                            jax.random.split(jax.random.PRNGKey(0), N), None, None, theta_is_vector=False)
  import ipdb; ipdb.set_trace()