from typing import Any, Optional, Sequence
from absl import flags

import math

import jax
import jax.numpy as jnp
import haiku
import gymnasium
import gymnasium.vector
import numpy as np

import functools
from src.utils import common
from src.gradient_estimators import gradient_estimator_utils
from src.task_parallelization import truncated_step
from src.task_parallelization import openai_gym_truncated_step
from src.utils import summary

FLAGS = flags.FLAGS

MetaParams = Any
TruncatedStep = Any
PRNGKey = Any

# this is different from vector_sample_perturbations
# as we don't need the positive and negatively perturbed thetas
# (theta, key, std) # only keys are different over multiple samples
sample_multiple_perturbations = jax.jit(jax.vmap(common.sample_perturbations, in_axes=(None, 0, None)))

################################# for RL ########################################

def test(
      i: int,
      theta: MetaParams, # a single theta
      smooth_train_eval_key: PRNGKey,
      non_smooth_test_key: PRNGKey,
      truncated_step_train_eval: openai_gym_truncated_step.OpenAIGymTruncatedStep,
      truncated_step_test: openai_gym_truncated_step.OpenAIGymTruncatedStep,
      summary_writer: summary.SummaryWriterBase,) -> None:
  """perform evaluation and log it through summary_writer
    if we don't want to run evaluation at iteration i, we just return None

  Args:
      i (int): the iteration number
      theta (MetaParams): the parameter to evaluate
      smooth_train_eval_key (PRNGKey): evaluation key for smoothed training loss
      non_smooth_test_key (PRNGKey): evaluation key for non-smoothed test loss
      truncated_step_train_eval (truncated_step.TruncatedStep): 
      truncated_step_test (truncated_step.TruncatedStep): 
      summary_writer (summary.SummaryWriterBase): a summary writer to log the evaluation results

  Returns:
      None
  """
  
  # check whether we should run evaluation at this iteration
  iteration_to_eval = i % FLAGS.eval_freq == 0
  if FLAGS.eval_watershed is not None:
    if i > FLAGS.eval_watershed:
      iteration_to_eval = i % FLAGS.second_eval_freq == 0

  if iteration_to_eval:
    if gradient_estimator_utils.smoothing_in_objective():
      # evaluate smoothed objective
      smoothed_train_reward, smoothed_train_reward_ci = evaluate_rl_reward(
        theta=theta,
        policy=truncated_step_train_eval.policy,
        env_name=FLAGS.gym_env_name,
        num_tasks=truncated_step_train_eval.num_tasks,
        key=smooth_train_eval_key,
        T=FLAGS.horizon_length,
        with_smoothing=True,
        sigma=FLAGS.sigma,
        K=FLAGS.K,
        analysis=False,
      )
      summary_writer.scalar("test/smoothed_train_reward", smoothed_train_reward, step=i)
      summary_writer.scalar("test/smoothed_train_reward_ci", smoothed_train_reward_ci, step=i)

    # evaluate non-smoothed objective
    unperturbed_test_reward, unperturbed_test_reward_ci =  evaluate_rl_reward(
      theta=theta,
      policy=truncated_step_train_eval.policy,
      env_name=FLAGS.gym_env_name,
      num_tasks=truncated_step_test.num_tasks,
      key=non_smooth_test_key,
      T=FLAGS.horizon_length,
      with_smoothing=False,
      sigma=FLAGS.sigma,
      K=FLAGS.K,
      analysis=False)

    summary_writer.scalar("test/unperturbed_test_reward", unperturbed_test_reward, step=i)
    summary_writer.scalar("test/unperturbed_test_reward_ci", unperturbed_test_reward_ci, step=i)
  return None


def evaluate_rl_reward(
      theta: MetaParams,
      policy: haiku.Transformed,
      env_name: str,
      num_tasks: int,
      key: PRNGKey,
      T: int,
      with_smoothing: bool = False,
      sigma: float = 0.01,
      K: Optional[int] = None,
      analysis=False):

  if not with_smoothing:
    sigma = 0.0
  if K is None:
    K = T

  # reward_list is of shape (horizon_length, n_particle)
  reward_list = full_horizon_reward_evaluate(
        theta,
        policy,
        env_name,
        num_tasks,
        key,
        sigma,
        K, # how frequent to get a new set of perturbations
    )

  if analysis:
    return np.array(reward_list)
  else:
    reward_over_particle = np.sum(reward_list, axis=0)
    return jnp.mean(reward_over_particle), 1.96 * jnp.std(reward_over_particle) / math.sqrt(reward_over_particle.shape[0])


@functools.partial(jax.jit, static_argnums=(0, 4))
def policy_vec_apply(policy, params, key, state, num_tasks):
  key_list = jax.random.split(key, num=num_tasks)
  return jax.vmap(policy.apply, in_axes=(0, 0, 0))(params, key_list, state)


def full_horizon_reward_evaluate(
      theta: MetaParams,
      policy: haiku.Transformed,
      env_name: str,
      num_tasks: int,
      key: PRNGKey,
      sigma: float,
      K: int,):
  init_key, sample_key = jax.random.split(key)
  rng = haiku.PRNGSequence(sample_key)  # type: ignore
  
  venv = gymnasium.vector.make(id=env_name, num_envs=num_tasks, asynchronous=False)
  np_seed = int(jax.random.randint(init_key, (), 0, 1e6))
  vobs, info = venv.reset(seed=np_seed)
  done = False
  t = 0 # t is for keeping track of the time step for sampling new noise

  reward_list = [] # will eventually be of shape (horizon, num_tasks)
  
  perturbed_multiple_thetas = None  # will be initialized in the first iteration
  while not done:
    need_resampling = ((t % K) == 0)
    if need_resampling:
      sample_key_list = jax.random.split(sample_key, num=num_tasks)
      new_epsilons = sample_multiple_perturbations(theta, sample_key_list, sigma)
      perturbed_multiple_thetas = \
        jax.tree_util.tree_map(lambda a,b: jnp.expand_dims(a, 0) + b, theta, new_epsilons)
    
    vaction = policy_vec_apply(
                policy, 
                perturbed_multiple_thetas, next(rng), vobs, num_tasks)
    vobs, vreward, vtermination, vtruncation, vinfo = venv.step(vaction)
    t += 1
    # print(f"t={t}\r")
    reward_list.append(vreward)

    done = np.any(vtermination) or np.any(vtruncation)
  
  return np.array(reward_list)

