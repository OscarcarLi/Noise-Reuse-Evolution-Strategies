from typing import Any, Optional, Sequence


from absl import flags
import math

import jax
import jax.numpy as jnp
import functools


from src.utils import common
from src.gradient_estimators import gradient_estimator_utils
from src.task_parallelization import truncated_step
from src.task_parallelization import dynamical_system_truncated_step
from src.utils import summary

FLAGS = flags.FLAGS

any_true = jax.jit(jnp.any)
MetaParams = Any
TruncatedStep = Any
PRNGKey = Any

# this is different from vector_sample_perturbations
# as we don't need the positive and negatively perturbed thetas
# (theta, key, std) # only keys are different over multiple samples
sample_multiple_perturbations = jax.jit(jax.vmap(common.sample_perturbations, in_axes=(None, 0, None)))

def test(
      i: int,
      theta: MetaParams, # a single theta
      smooth_train_eval_key: PRNGKey,
      non_smooth_test_key: PRNGKey,
      truncated_step_train_eval: truncated_step.TruncatedStep,
      truncated_step_test: truncated_step.TruncatedStep,
      summary_writer: summary.SummaryWriterBase,) -> None:
  """perform evaluation and log it through summary_writer
    if we don't want to run evaluation at iteration i, we just return None
    The evaluation frequency is governed by FLAGS values defined in main_training.py

  Args:
      i (int): the iteration number
      theta (MetaParams): the parameter to evaluate
      smooth_train_eval_key (PRNGKey): evaluation key for smoothed training loss
      non_smooth_test_key (PRNGKey): evaluation key for non-smoothed test loss
      truncated_step_train_eval (truncated_step.TruncatedStep)
      truncated_step_test (truncated_step.TruncatedStep)
        NOTE: both of the TruncatedStep objects above need to be jax jittable.
        # subclass object like OpenAIGymTruncatedStep doesn't work with this.
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
    if (
      (gradient_estimator_utils.smoothing_in_objective()) 
          and (FLAGS.application.upper() != "lopt".upper())
    ):
      # we only evaluate smoothed training loss when the methods use smoothing
      smoothed_train_loss, smoothed_train_loss_ci = test_through_truncatedstep(
        theta=theta,
        truncated_step=truncated_step_train_eval,
        key=smooth_train_eval_key,
        with_smoothing=True,
        sigma=FLAGS.sigma,
        K=FLAGS.K,
      )
      summary_writer.scalar("test/smoothed_meta_train_loss", smoothed_train_loss, step=i)
      summary_writer.scalar("test/smoothed_meta_train_loss_ci", smoothed_train_loss_ci, step=i)

    # we always run unperturbed test evaluation if run_test
    unperturbed_test_loss, unperturbed_test_loss_ci = test_through_truncatedstep(
      theta=theta,
      truncated_step=truncated_step_test,
      key=non_smooth_test_key,
      with_smoothing=False,
      sigma=FLAGS.sigma,
      K=FLAGS.K,
    )
    summary_writer.scalar("test/unperturbed_meta_test_loss", unperturbed_test_loss, step=i)
    summary_writer.scalar("test/unperturbed_meta_test_loss_ci", unperturbed_test_loss_ci, step=i)
  return None

def test_through_truncatedstep(
      theta: MetaParams,
      truncated_step: truncated_step.TruncatedStep,
      key: PRNGKey,
      with_smoothing: bool = False,
      sigma: float = 0.01,
      K: Optional[int] = None,
      analysis=False):

  T = truncated_step.T
  if not with_smoothing:
    sigma = 0.0
  data_batch_list = truncated_step.get_batch(steps=T)
  if K is None:
    K = T

  # losses is of shape (horizon_length, n_particle)
  losses = full_horizon_loss_evaluate(
        theta,
        truncated_step, # use large number of num_tasks, use constant truncation schedule,
        key,
        sigma,
        K, # how frequent to get a new set of perturbations
        data_batch_list, # a pytree with each leaf node having horizon_length * n_particles
    )

  if analysis:
    return jnp.asarray(losses)
  else:
    loss_over_particle = jnp.mean(jnp.asarray(losses), axis=0)
    return jnp.mean(loss_over_particle), 1.96 * jnp.std(loss_over_particle) / math.sqrt(loss_over_particle.shape[0])


@functools.partial(
  jax.jit, static_argnums=(1,)
)
def full_horizon_loss_evaluate(
        theta: MetaParams,
        truncated_step: truncated_step.TruncatedStep, # use large number of num_tasks, use constant truncation schedule,
        key: PRNGKey,
        sigma: float,
        K: int, # how frequent to get a new set of perturbations
        data_batch_list: Any, # a pytree with each leaf node having horizon_length * n_particles
    ) -> jax.Array:
  """Evaluate the loss by iterating through batches of data given in data_batch_list


  Args:
      theta (MetaParams): the unrolling parameter
      truncated_step (TruncatedStep): the object that manages the multiple particle
        dynamical system init and unrolling.
      key (PRNGKey): randomness key for the noise sampling and unrolls
      sigma (float): noise sampling frequency
      K (int): how many steps to sample a new noise
      data_batch_list (Any): a list of data batches. When using `DynamicalSystemTruncatedStep`
        the object is simply an array full of None as `DynamicalSystem` internally
        keeps track of its data

  Returns:
      jax.Array: losses of shape (horizon_length, n_particles)
        one loss for each time step for each particle.
  """
  def step(carry, data_batch):
    unroll_states, perturbed_multiple_thetas, t, key = carry
    key, unroll_key, sample_key = jax.random.split(key, 3)
    next_unroll_states, unroll_states_out = \
      truncated_step.unroll_step(
          perturbed_multiple_thetas,
          unroll_states,
          jax.random.split(unroll_key, truncated_step.num_tasks),
          data_batch,
          None, # outer_state
          True, # theta_is_vector
      )
    def true_fn():
      sample_key_list = jax.random.split(sample_key, num=truncated_step.num_tasks)
      new_epsilons = sample_multiple_perturbations(theta, sample_key_list, sigma)
      new_perturbed_multiple_thetas = \
        jax.tree_util.tree_map(lambda a,b: jnp.expand_dims(a, 0) + b, theta, new_epsilons)
      return new_perturbed_multiple_thetas
    def false_fn():
      return perturbed_multiple_thetas

    need_resampling = ((t + 1) % K) == 0
    next_perturbed_multiple_thetas = jax.lax.cond(need_resampling, true_fn, false_fn)

    return (next_unroll_states, next_perturbed_multiple_thetas, t+1, key), \
            unroll_states_out.loss
  
  ### setting the initial carry ###
  init_key, sample_key, step_key = jax.random.split(key, 3)
  # keys for sampling the perturbations
  sample_key_list = jax.random.split(sample_key, truncated_step.num_tasks)
  epsilons = sample_multiple_perturbations(theta, sample_key_list, sigma)
  # p for perturbed
  p_multiple_thetas = jax.tree_util.tree_map(lambda a,b: jnp.expand_dims(a, 0) + b, theta, epsilons)
  unroll_states = truncated_step.init_step_state(
    theta=p_multiple_thetas,
    outer_state=None, # this only tracks outer loop iteration number
    key_list=jax.random.split(init_key, truncated_step.num_tasks),
    theta_is_vector=True)
  
  ### compute the losses ###
  _, losses = jax.lax.scan(step, (unroll_states, p_multiple_thetas, 0, step_key), data_batch_list)

  # losses of shape (horizon_length, n_particles)
  return losses





