from absl import flags

import math
import jax

from src.task import lorenz
from src.task_parallelization import dynamical_system_truncated_step
from src.gradient_estimators import gradient_estimator_utils

FLAGS = flags.FLAGS

# problem type
flags.DEFINE_float("lorenz_dt", 0.005, "time increment to run forward integration on")
flags.DEFINE_float("lorenz_init_loga", None, "initial value of a, this will be inverse transformed to its internal representation")
flags.DEFINE_float("lorenz_init_logr", None, "initial value of r, this will be inverse transformed to its internal representation")
flags.DEFINE_boolean("lorenz_single_init", False, "whether to use a single inner init or not")

def create_task_train_test():
  if FLAGS.lorenz_single_init:
    train_task = lorenz.Lorenz_loga_logr_DynamicalSystem_FixedInit(
      dt=FLAGS.lorenz_dt,
      T=FLAGS.horizon_length,
      ground_truth_a=10.0, # the typically used value
      ground_truth_r=28.0, 
      ground_truth_b=8.0/3.0,
      init_a=math.exp(FLAGS.lorenz_init_loga),
      init_r=math.exp(FLAGS.lorenz_init_logr),
    )
  else:
    train_task = lorenz.Lorenz_loga_logr_DynamicalSystem_FiniteGaussianInit(
      dt=FLAGS.lorenz_dt,
      T=FLAGS.horizon_length,
      ground_truth_a=10.0, # the typically used value
      ground_truth_r=28.0, 
      ground_truth_b=8.0/3.0,
      init_a=math.exp(FLAGS.lorenz_init_loga),
      init_r=math.exp(FLAGS.lorenz_init_logr),
      inner_init_sigma=1.0,
      n_init=100,
      key=jax.random.PRNGKey(0),
    )
  test_task = lorenz.Lorenz_loga_logr_DynamicalSystem_InfiniteGaussianInit(
    dt=FLAGS.lorenz_dt,
    T=FLAGS.horizon_length,
    ground_truth_a=10.0, # the typically used value
    ground_truth_r=28.0, 
    ground_truth_b=8.0/3.0,
    init_a=math.exp(FLAGS.lorenz_init_loga),
    init_r=math.exp(FLAGS.lorenz_init_logr),
    inner_init_sigma=1.0, # what is a good value for this?
  )
  return train_task, test_task

def create_truncated_step_train_test(train_task, test_task, num_test_particles=10000):
  # return a tuple of (truncated_step_train, truncated_step_test)

  # only use random initial iteration if using truncation methods
  # if using full trajectory, always start from the beginning (not using random initial iteration)
  random_initial_iteration_train = \
    gradient_estimator_utils.use_random_initial_iteration_for_truncated_step_train()
  truncated_step_train = \
    dynamical_system_truncated_step.DynamicalSystemDecomposableTruncatedStepMultiDevice(
      dynamical_system=train_task,
      num_tasks=(FLAGS.num_particles // FLAGS.num_gradient_estimators),
      T=FLAGS.horizon_length,
      random_initial_iteration=random_initial_iteration_train,
      truncation_window_size=FLAGS.trunc_length,
      # for full trajectory methods like FullES and FullGradient
      # the random_initial_iteration will be turned off so even though
      # truncation_window_size is set to arbitrary values it doesn't matter
      device_list=jax.devices(),
    )

  truncated_step_test = \
    dynamical_system_truncated_step.DynamicalSystemDecomposableTruncatedStep(
      dynamical_system=test_task,
      num_tasks=num_test_particles, # probably need to set an appropriate number depending on the variance of the dataset
      T=FLAGS.horizon_length,
      random_initial_iteration=False,
    )

  return (truncated_step_train, truncated_step_test)

def create_truncated_step_train_for_evaluation(train_task, num_particles=10000):
  truncated_step_train_evaluation = \
    dynamical_system_truncated_step.DynamicalSystemDecomposableTruncatedStep(
      dynamical_system=train_task,
      num_tasks=num_particles,
      T=FLAGS.horizon_length,
      random_initial_iteration=False,
      truncation_window_size=FLAGS.trunc_length,
      # for full trajectory methods like FullES and FullGradient
      # the random_initial_iteration will be turned off so even though
      # truncation_window_size is set to arbitrary values it doesn't matter
      # device_list=jax.devices(),
    )
  return truncated_step_train_evaluation
  