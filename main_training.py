# coding=utf-8
"""Unrolled computation graph training using gradient estimator APIs."""
# again, this only works on startup!
USE_64 = False
from jax import config
config.update("jax_enable_x64", USE_64)

import tensorflow as tf
tf.config.experimental.set_visible_devices([], "GPU")

from typing import Any, Sequence

from absl import app, logging
from absl import flags
import jax
import jax.numpy as jnp
from src.utils import filesystem
from src.utils import summary
from src.optimizers import base as opt_base
from src.outer_trainers import gradient_learner
from src.gradient_estimators import gradient_estimator_utils

# from src.applications import dd_utils
from src.applications import lopt_utils
from src.applications import lorenz_utils
from src.applications import rl_utils

import src.optimizers.optimizers_utils as optimizers_utils

from src.eval import evaluation
from src.eval import evaluation_rl
from flax.training import checkpoints
import numpy as np
import tqdm
import pytz
from datetime import datetime


FLAGS = flags.FLAGS

logging.set_verbosity(logging.ERROR)

############################################################
############## application specification ###################
############################################################
# for the specific application, need to go to the corresponding utils file
# to look for the other configs
flags.DEFINE_string("application", None, "the type of application")
flags.DEFINE_integer("horizon_length", None,
                     "the length of one full trajectory, T in the paper",)

############################################################
####### gradient estimation method specification ###########
############################################################
flags.DEFINE_string("gradient_estimator", "TruncatedES", "the name of the gradient estimator to use")
flags.DEFINE_integer("num_particles", None, 
                    "number of particles, N in the paper")
flags.DEFINE_integer("num_gradient_estimators", 1, 
                    ("this number should typically be set to 1"
                    "this is not to be confused with the number of particles,"
                    "when set to a larger than 1 number, multiple sets of parallel"
                    "workers will be used to compute the gradient estimator"
                    "one set at a time"
                    "this essentially is the same as microbatching"
                    ))
flags.DEFINE_integer("trunc_length", None,
                     "the length of the truncation window")
flags.DEFINE_integer("K", None, "how many steps is the noise in PES reused")
flags.DEFINE_float("sigma", None, "Gaussian smoothing std")
flags.DEFINE_boolean("loss_normalize", False, "whether to normalize the gradient by the loss std, only supported by FullES, TruncatedPESK")

############################################################
######         meta-optimization specification  ############
############################################################
flags.DEFINE_string("outer_optimizer_specs", None,
                    ("a string specifying the first order meta-optimization"
                     "algorithm and learning rate")
)
flags.DEFINE_integer("outer_iterations", None, "number of meta-updates")
flags.DEFINE_integer("init_seed", None, "the seed to determine the initialized meta-parameter theta")
flags.DEFINE_integer("remaining_seed", None, "the seed for rest of the unroll")
flags.DEFINE_float("grad_clip_threshold", None, "the coordinate clipping threshold for each coordinate of gradient")
flags.DEFINE_boolean("run_test_during_training", False, "whether to run test evaluation or not throughout training")
flags.DEFINE_boolean("run_test_after_training", False, "whether to run test evaluation or not after training is finished")

############################################################
###### evaluation specification ############################
############################################################
flags.DEFINE_integer("eval_freq", None, "how many outer iterations to run a test evaluations, only used if run_test is True")
flags.DEFINE_integer("eval_watershed", None, "after this many outer iterations, we start to evaluate every second_eval_freq iterations")
flags.DEFINE_integer("second_eval_freq", None, "how frequent to run a test evaluations after eval_watershed, only used if run_test is True")

MetaParams = Any
PRNGKey = Any

def tree_zip_jnp(xs):
  xs = list(xs)
  _, tree_def = jax.tree_util.tree_flatten(xs[0])
  ys = map(jnp.asarray, zip(*map(lambda x: jax.tree_util.tree_flatten(x)[0], xs)))
  return jax.tree_util.tree_unflatten(tree_def, ys)


def tree_zip_onp(xs):
  xs = list(xs)
  _, tree_def = jax.tree_util.tree_flatten(xs[0])
  ys = map(np.asarray, zip(*map(lambda x: jax.tree_util.tree_flatten(x)[0], xs)))
  return jax.tree_util.tree_unflatten(tree_def, ys)


def train():
  """Learn the parameter of a dynamical system"""
  print("#" * 80)
  print("run test after training:", FLAGS.run_test_after_training)
  if FLAGS.run_test_during_training:
    FLAGS.run_test_after_training = False

  if FLAGS.remaining_seed:
    # this will be logged in log dir name
    remaining_seed = FLAGS.remaining_seed
  else:
    remaining_seed = np.random.randint(0, int(2**30)) % 100

  assert FLAGS.num_particles % FLAGS.num_gradient_estimators == 0

  # test K is None for none PESK methods
  # this is to ensure we are evaluating the method correctly
  if "K" in FLAGS.gradient_estimator.upper():
    assert FLAGS.K is not None

  # this is the outer loop optimizer
  theta_opt = optimizers_utils.create_optimizer(FLAGS.outer_optimizer_specs)
  if FLAGS.grad_clip_threshold is not None:
    print(f"applying coordinate gradient clipping {FLAGS.grad_clip_threshold}")
    theta_opt = opt_base.GradientClipOptimizer(theta_opt, grad_clip=FLAGS.grad_clip_threshold)
  print(theta_opt)

  # create the task for the specific application
  if FLAGS.application.upper() == "lopt".upper():
    application_utils = lopt_utils
    evaluation_utils = evaluation
  elif FLAGS.application.upper() == "lorenz".upper():
    application_utils = lorenz_utils
    evaluation_utils = evaluation
  elif FLAGS.application.upper() == "rl".upper():
    application_utils = rl_utils
    evaluation_utils = evaluation_rl
  else:
    assert False, f"application {FLAGS.application} not supported"

  # we need task because we need to pass it for the init
  train_task, test_task = application_utils.create_task_train_test()
  truncated_step_train, truncated_step_test = \
    application_utils.create_truncated_step_train_test(train_task, test_task)
  truncated_step_train_evaluation = \
    application_utils.create_truncated_step_train_for_evaluation(train_task)

  grad_est = gradient_estimator_utils.create_gradient_estimator(truncated_step_train)
    
  # problem type
  train_log_dir = f"./runs/{FLAGS.application}{'_64' if USE_64 else ''}/"
  train_log_dir += f"{train_task.name()}/"

  # add the name of the gradient estimator
  if FLAGS.num_gradient_estimators > 1:
    train_log_dir += f"({grad_est.grad_est_name()})x{FLAGS.num_gradient_estimators}"
  else:
    train_log_dir += f"{grad_est.grad_est_name()}"
  # rest of shared naming for train_log_dir
  train_log_dir += f"_initseed{FLAGS.init_seed}remainingseed{remaining_seed}_outeropt{FLAGS.outer_optimizer_specs}_outeriterations{FLAGS.outer_iterations}_{'runtest' if (FLAGS.run_test_during_training or FLAGS.run_test_after_training) else 'notest'}"
  timezone = pytz.timezone("America/New_York")
  train_log_dir += "_" + datetime.now(timezone).strftime("%d:%b:%Y:%H:%M:%S")

  gradient_estimators = [grad_est for _ in range(FLAGS.num_gradient_estimators)]

  outer_trainer = gradient_learner.SingleMachineGradientLearner(
      train_task, gradient_estimators, theta_opt)

  filesystem.make_dirs(train_log_dir)
  summary_writer = summary.MultiWriter(
      summary.JaxboardWriter(train_log_dir), summary.PrintWriter())

  theta_init_key = jax.random.PRNGKey(FLAGS.init_seed)
  remaining_key = jax.random.PRNGKey(remaining_seed)
  grad_est_init_key, key = jax.random.split(remaining_key,)
  outer_trainer_state = outer_trainer.init(
    theta_init_key=theta_init_key, grad_est_init_key=grad_est_init_key)

  # meta-training
  theta_trajectory = []
  losses = [] # periodically refreshed
  for i in tqdm.trange(0, FLAGS.outer_iterations + 1):
    # save the meta-parameter theta
    # checkpoints.save_checkpoint(ckpt_dir=train_log_dir, prefix="theta", value=theta, step=i) 
    key3, key2, key1, key = jax.random.split(key, num=4)
    
    if i > 0: # the first step only perform checkpoint saving and evaluation
      with_m = True if i % 10 == 0 else False
      outer_trainer_state, loss, metrics = outer_trainer.update(
          outer_trainer_state, key1, with_metrics=True)
      losses.append(loss)

      if with_m:
        # we only use smoothing over meta loss
        summary_writer.scalar("average_meta_loss", np.mean(losses), step=i)
        losses = []
      
      # we always log the summary statistics for every iteration
      # log out summaries to tensorboard
      for k, v in metrics.items():
        agg_type, metric_name = k.split("||")
        if agg_type == "collect":
          summary_writer.histogram(metric_name, v, step=i)
        else:
          summary_writer.scalar(metric_name, v, step=i)
      summary_writer.flush()

    # log the meta-parameters's progression
    if FLAGS.application.upper() == "lorenz".upper():
      for mp_name, mp_val in train_task.meta_parameter_to_dict(
        outer_trainer_state.gradient_learner_state.theta_opt_state.params).items():
        summary_writer.scalar(f"{train_task.name()}/{mp_name}", np.array(mp_val), step=i)
    
    theta = outer_trainer.gradient_learner.get_meta_params(
        outer_trainer_state.gradient_learner_state)
    theta_trajectory.append(theta)
    if FLAGS.application.upper() == "rl".upper():
      checkpoints.save_checkpoint(
        ckpt_dir=train_log_dir,
        target=theta,
        step=i,
        prefix="theta",
        keep=1000001,
        overwrite=False)

    # count how many training steps is used
    summary_writer.scalar("total_env_steps", grad_est.total_env_steps_used, step=i)

    if FLAGS.run_test_during_training:
      evaluation_utils.test(
        i=i,
        theta=theta,
        smooth_train_eval_key=key2,
        non_smooth_test_key=key3,
        truncated_step_train_eval=truncated_step_train_evaluation, # type: ignore
        truncated_step_test=truncated_step_test, # type: ignore
        summary_writer=summary_writer,
      )

  ###### training finished #######
  # save the entire trajectory of theta
  theta_trajectory_pytree = tree_zip_onp(theta_trajectory)
  checkpoints.save_checkpoint(
    ckpt_dir=train_log_dir,
    target=theta_trajectory_pytree,
    step=0,
    prefix="theta_trajectory",
    keep=1000001,
    overwrite=False)

  if FLAGS.run_test_after_training:
    _, key = jax.random.split(remaining_key,)
    # meta-test after everyting finished training
    for i in tqdm.trange(0, FLAGS.outer_iterations + 1):
      # we still unpack the keys the same, this would make it the same as running
      # the test evaluation sequentially interleaved with the training
      key3, key2, key1, key = jax.random.split(key, num=4)
      theta = jax.tree_util.tree_map(lambda x: x[i], theta_trajectory_pytree)
      evaluation_utils.test(
        i=i,
        theta=theta,
        smooth_train_eval_key=key2,
        non_smooth_test_key=key3,
        truncated_step_train_eval=truncated_step_train_evaluation, # type: ignore
        truncated_step_test=truncated_step_test, # type: ignore
        summary_writer=summary_writer,)

  return None


def main(unused_argv: Sequence[str]) -> None:
  train()


if __name__ == "__main__":
  app.run(main)
