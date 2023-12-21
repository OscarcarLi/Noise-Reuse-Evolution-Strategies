import gymnasium as gym
import numpy as np
import copy
from typing import Tuple, Dict, Any

from src.utils.mujoco_env_utils import (
  MUJOCO_ENV_NAMES, EnvCreator, copy_mujoco_env, copy_mujoco_sync_venv)

T = 1000

def get_single_env_unroll_outputs(env, action_sequences):
  # unroll 
  obs_through_time = []
  r_through_time = []
  terminated_through_time = []
  truncated_through_time = []
  info_through_time = []
  # for now we skip logging info
  for i in range(action_sequences.shape[0]):
    obs, reward, terminated, truncated, info = env.step(action_sequences[i])
    if terminated or truncated:
      # this would replace the original obs if an episode resets
      obs, info = env.reset()
    obs_through_time.append(obs)
    r_through_time.append(reward)
    terminated_through_time.append(terminated)
    truncated_through_time.append(truncated)
    info_through_time.append(info)


  obs_through_time = np.array(obs_through_time)
  r_through_time = np.array(r_through_time)
  terminated_through_time = np.array(terminated_through_time)
  truncated_through_time = np.array(truncated_through_time)
  return {
    "obs": obs_through_time,
    "reward": r_through_time,
    "terminated": terminated_through_time,
    "truncated": truncated_through_time,
    "info": truncated_through_time,
  }

def single_env_copying_compare(env_name, root_seed, copy_fn) -> Tuple[bool, Dict[str, Any]]:
  # preparing an env to be copied
  np.random.seed(root_seed)
  env = gym.make(env_name)
  env_seed = np.random.randint(0, 100000)
  obs, info = env.reset(seed=env_seed)
  copy_time_step = np.random.randint(low=0, high=T)
  preparation_action_sequences = np.random.normal(
    size=[copy_time_step] + list(env.action_space.shape))
  for i in range(copy_time_step):
    env.step(preparation_action_sequences[i])
    
  env_copy = copy_fn(env)
  action_sequences = np.random.normal(size=[T] + list(env.action_space.shape))

  result_dict = get_single_env_unroll_outputs(
    env, action_sequences=action_sequences)
  result_dict_copy = get_single_env_unroll_outputs(
    env_copy, action_sequences=action_sequences)
  
  # ipdb.set_trace()
  info = {"env_dict": result_dict, "env_copy_dict": result_dict_copy,}
  for key in result_dict.keys():
    # EPSILON = 1e-7
    # should we use some threshold for checking?
    if not np.array_equal(result_dict[key], result_dict_copy[key]):
      # ipdb.set_trace()
      info["failed"] = key
      return False, info

  return True, info



def get_venv_unroll_outputs(venv, action_sequences):
  # unroll 
  obs_through_time = []
  r_through_time = []
  terminated_through_time = []
  truncated_through_time = []
  # for now we skip logging info
  for i in range(action_sequences.shape[0]):
    obs, reward, terminated, truncated, info = venv.step(action_sequences[i])
    obs_through_time.append(obs)
    r_through_time.append(reward)
    terminated_through_time.append(terminated)
    truncated_through_time.append(truncated)

  obs_through_time = np.array(obs_through_time)
  r_through_time = np.array(r_through_time)
  terminated_through_time = np.array(terminated_through_time)
  truncated_through_time = np.array(truncated_through_time)
  return {
    "obs": obs_through_time,
    "reward": r_through_time,
    "terminated": terminated_through_time,
    "truncated": truncated_through_time,
  }

def venv_copying_compare(env_name, num_particles, root_seed, venv_copy_fn):
  np.random.seed(root_seed)
  
  # preparing a venv to be copied
  time_step_list = np.random.randint(low=0, high=T, size=(num_particles,))
  env_list = []
  for i in range(num_particles):
    env = gym.make(env_name)
    seed = np.random.randint(low=0, high=int(1e6))
    obs, info = env.reset(seed=int(seed))

    for _ in range(time_step_list[i]):
      env.step(np.random.normal(size=env.action_space.shape))
    env_list.append(env)

  env_fns = [EnvCreator(env=env) for env in env_list]
  venv = gym.vector.SyncVectorEnv(env_fns=env_fns)
  venv_copy = venv_copy_fn(venv)
  
  action_sequences = np.random.normal(size=[T] + list(venv.action_space.shape))

  result_dict = get_venv_unroll_outputs(
    venv=venv, action_sequences=action_sequences)
  result_dict_copy = get_venv_unroll_outputs(
    venv=venv_copy, action_sequences=action_sequences)
  
  info = {"venv_dict": result_dict, "venv_copy_dict": result_dict_copy,}
  for key in result_dict.keys():
    if not np.array_equal(result_dict[key], result_dict_copy[key]):
      info["failed"] = key
      return False, info

  return True, info

def test_single_env_copying(copy_fn, n_tests_per_env=10):
  print(copy_fn.__name__)
  for env_name in MUJOCO_ENV_NAMES:
    count = 0
    for i in range(1, n_tests_per_env + 1):
      seed = np.random.randint(0, 100000)
      match, info = \
        single_env_copying_compare(env_name=env_name, root_seed=seed, copy_fn=copy_fn)
      if match:
        count += 1
      print("\t", env_name, f"Success: {count}/{i}", end="\r")
    print()

def test_venv_copying(venv_copy_fn, n_tests_per_env=10):
  venv_copy_fn = copy_mujoco_sync_venv
  print(venv_copy_fn.__name__)
  for env_name in MUJOCO_ENV_NAMES:
    count = 0
    for i in range(1, n_tests_per_env + 1):
      seed = np.random.randint(low=0, high=int(1e6))

      match, info = \
        venv_copying_compare(
          env_name=env_name, num_particles=10, root_seed=seed, venv_copy_fn=venv_copy_fn)
      if match:
        count += 1
      print("\t", env_name, f"Success: {count}/{i}", end="\r")
    print()

if __name__ == "__main__":
  # test_single_env_copying(copy_fn=copy_mujoco_env)
  test_venv_copying(venv_copy_fn=copy_mujoco_sync_venv)