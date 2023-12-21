import gymnasium as gym
import numpy as np
import copy
import time

MUJOCO_ENV_NAMES = [
  "Ant-v4",
  "HalfCheetah-v4",
  "Hopper-v4",
  "HumanoidStandup-v4",
  "Humanoid-v4",
  "InvertedDoublePendulum-v4",
  "InvertedPendulum-v4",
  "Reacher-v4",
  "Swimmer-v4",
  "Pusher-v4",
  "Walker2d-v4",
]

class EnvCreator:
  """
  a callable object that returns the env
  this resolves the issue of using lambda expressions for creating vector env
  (venv) because lambda expressions are only evaluated at run time
  """
  def __init__(self, env):
    self._env = env
  def __call__(self):
    return self._env

def copy_mujoco_env(env):
  """
    env is a mujoco env (with wrappers) produced by gym.make("Ant-v4")
    return a deep copy of the env so that stepping env and env_copy result
    in exactly the same obs, rewards, truncated, and terminated output
  """

  # deepcopy works through __reduce_ex call for env.env.env.env (for example Ant-v4 object)
  # https://peps.python.org/pep-0307/
  # implicitly this relies on a pair of methods defined by the object
  # __setstate__ and __getstate__
  # because env.env.env.env inherits from EzPickle
  # https://github.com/Farama-Foundation/Gymnasium/blob/e0994f028ab9a88f6c8c1afa9e75f6517b1c1c1e/gymnasium/utils/ezpickle.py#L5
  # it overrides the default object's __setstate__ and __getstate__
  # by doing this, it doesn't __getstate__ the .data and .np_random generator 
  # thus after this deepcopy, we still need to manager those two attributes so that it works correctly
  env_copy = copy.deepcopy(env)
  """
  To illustrate how the simulation state can be manipulated, suppose we have two mjData pointers src and dst corresponding to the same mjModel, and we want to copy the entire simulation state from one to the other (leaving out internal diagnostics which do not affect the simulation). This can be done as
  // copy simulation state
  dst->time = src->time;
  mju_copy(dst->qpos, src->qpos, m->nq);
  mju_copy(dst->qvel, src->qvel, m->nv);
  mju_copy(dst->act,  src->act,  m->na);

  // copy mocap body pose and userdata
  mju_copy(dst->mocap_pos,  src->mocap_pos,  3*m->nmocap);
  mju_copy(dst->mocap_quat, src->mocap_quat, 4*m->nmocap);
  mju_copy(dst->userdata, src->userdata, m->nuserdata);

  // copy warm-start acceleration
  mju_copy(dst->qacc_warmstart, src->qacc_warmstart, m->nv);
  """
  # without this line below, env_copy.data.body("torso") would be all zeros
  # we need this for the first step reward evaluation for envs like Ant-v4
  # otherwise they wouldn't match
  # in addition, we cannot simply perform left hand assignment
  # through env_copy.data (because this would add a data field in the wrapper class
  # however, the .get_body_com is only defined on the unwrapped class,
  # which would then query .data field of the unwrapped class
  # which doesn't have this updated deepcopy
  # https://github.com/deepmind/mujoco/blob/15e77b13dd87e1f28210d109a8de7dff3583999d/python/mujoco/structs.cc#L1791
  handle = env_copy
  while isinstance(handle, gym.Wrapper):
    handle = handle.env
  handle.data = copy.deepcopy(env.data)

  # the random generator needs to also be synced up
  # here this assignment would make env.env.env.env.np_random also updated
  # this is because np_random is a property and it has a setter method
  # https://github.com/Farama-Foundation/Gymnasium/blob/e0994f028ab9a88f6c8c1afa9e75f6517b1c1c1e/gymnasium/core.py#L211
  # this is different from env_copy.env.env.env.data, where if you just do env_copy.data = copy.deepcopy(env.data)
  # it wouldn't get updated.
  env_copy.np_random = copy.deepcopy(env.np_random)
  return env_copy

def copy_mujoco_sync_venv(venv):
  """copy a vector env of mujoco envs

  Args:
      venv (gym.vector.sync_vector_env.SyncVectorEnv): a vector env of mujoco envs (synchronous)
  Returns:
      gym.vector.sync_vector_env.SyncVectorEnv: a vector env of mujoco envs that
        behaves exactly the same as the input venv
  """
  start_copying_time = time.time()
  env_list = venv.envs
  env_copy_list = []
  for env in env_list:
    env_copy = copy_mujoco_env(env)
    env_copy_list.append(env_copy)
  end_individual_copying_time = time.time()
  # print("time spent on individual copying: ", end_individual_copying_time - start_copying_time)
  
  env_copy_fns = [EnvCreator(env=env_copy) for env_copy in env_copy_list]
  venv_copy = gym.vector.SyncVectorEnv(env_copy_fns)
  end_copying_time = time.time()
  # print("time spent on creating venv: ", end_copying_time - end_individual_copying_time)
  return venv_copy