"""
Check the vector env deep copying for immutable states needed by TruncatedESBiased
python3 -m src.task_parallelization.openai_gym_truncated_step_test
"""

import gymnasium as gym
import numpy as np
import copy

import ipdb
import pickle

from src.task_parallelization import openai_gym_truncated_step
from src.utils.mujoco_env_utils import (copy_mujoco_sync_venv)

# ENV_NAME = "Swimmer-v4"
ENV_NAME = "HalfCheetah-v4"
NUM_PARTICLES = 100
SEED = 42
T = 1000

np.random.seed(SEED)
time_step_list = np.random.randint(low=0, high=T, size=(NUM_PARTICLES,))
env_list = []
for i in range(NUM_PARTICLES):
  env = gym.make(ENV_NAME)
  seed = np.random.randint(low=0, high=int(1e6), size=tuple())
  obs, info = env.reset(seed=int(seed))

  for _ in range(time_step_list[i]):
    env.step(np.random.normal(size=env.action_space.shape))
  env_list.append(env)

env_fns = [openai_gym_truncated_step.EnvCreator(env=env) for env in env_list]

venv = gym.vector.SyncVectorEnv(env_fns=env_fns)
print("copying venv")
venv_copy = openai_gym_truncated_step.copy_mujoco_sync_venv(venv)

action_sequences = np.random.normal(size=[T] + list(venv.action_space.shape)) # type: ignore

print("stepping venv")
vobs_list = []
for i in range(T):
  vobs, vreward, vterminated, vtruncated, vinfo = venv.step(action_sequences[i])
  vobs_list.append(vobs)
vobs_list = np.array(vobs_list)

print("stepping venv_copy")
vobs_list_copy = []
for i in range(T):
  vobs_copy, vreward_copy, vterminated_copy, vtruncated_copy, vinfo_copy = venv_copy.step(action_sequences[i])
  vobs_list_copy.append(vobs_copy)
vobs_list_copy = np.array(vobs_list_copy)

diff = vobs_list - vobs_list_copy
print(np.sum(np.absolute(diff), axis=(0, 2)))
ipdb.set_trace()