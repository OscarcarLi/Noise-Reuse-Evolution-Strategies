from absl import flags

import tensorflow as tf
import tensorflow_datasets as tfds

from typing import Sequence, Any, Optional, Callable, Tuple

import jax
import jax.numpy as jnp
import haiku
import optax._src.loss

from src.task import learned_optimizer as learned_optimizer_lib
from src.task import learned_optimizer_inner_models
from src.learned_optimizers import base as lopt_base
from src.learned_optimizers import mlp_lopt
from src.utils import common


from src.task_parallelization import dynamical_system_truncated_step
from src.gradient_estimators import gradient_estimator_utils

FLAGS = flags.FLAGS

# problem type
flags.DEFINE_string("lopt_dataset_name", None,
                    ("the name of the dataset to use",
                    "currently support fashion_mnist, cifar10, cifar100"),)
flags.DEFINE_integer("image_length", None, "the size of the image",) # a single integer and we make the image square
flags.DEFINE_integer("lopt_batch_size", 128, "number of examples in a train/val batch",) # a single integer and we make the image square
flags.DEFINE_boolean("use_val", True, "whether to use validation set (True) or training set (False) for outer loss computation",)
flags.DEFINE_boolean("use_accuracy", False, "whether to use accuracy or cross entropy loss as outer loss",)
flags.DEFINE_boolean("use_threshold_outer_cross_entropy_loss", False,
                    """only relevant for cross_entropy loss (use_accuracy=False),
                    threshold the per example cross_entropy loss to be at most 1.5 * ln(num_classes)""")

flags.DEFINE_string("lopt_inner_model", None, 
                    ("the inner model to use"
                    "currently supporting mlp and cnn"),)
flags.DEFINE_string("lopt_activation", "relu", "the activation function to use",)
flags.DEFINE_string("lopt_hidden_dims", "32", "the comma separated hidden dimensions of the MLP which doesn't count the class dimension",)
flags.DEFINE_string("lopt_hidden_channels", None, "the comma separated hidden channels of the CNN which doesn't count the class dimension",)

flags.DEFINE_string("lopt_architecture", "LearnableSGDM", "the class of learned optimizer",)
flags.DEFINE_integer("lopt_mlplopt_hidden_size", None, "the hidden size of the MLPLOpt",)
flags.DEFINE_integer("lopt_mlplopt_num_hidden_layers", None, "the number of hidden layers of the MLPLOpt",)

flags.DEFINE_integer("lopt_single_sequence_seed", None, "the seed for using a single sequence with deterministic inner init and unroll",)

PRNGKey = Any
Params = Any
# this is the inner model's state for prediction (somethings like exponential moving average, batch stats)
State = Any

DATASET_TO_NUM_CLASSES = {
  "FASHION_MNIST": 10,
  "CIFAR10": 10,
  "CIFAR100": 100,
}

def get_entire_Xy_tuple_reshape(
  dataset_name: str,
  split: str,
  image_shape: Optional[Sequence[int]]=None,) -> tuple[jax.Array, jax.Array]:
  X, y = tfds.as_numpy(tfds.load(
                          dataset_name,
                          split=split,
                          shuffle_files=False,
                          batch_size=-1,
                          as_supervised=True,)) # type: ignore
  if image_shape is not None:
    X = tf.image.resize(X, image_shape)

  X = jnp.array(X / 255.0) # type: ignore
  y = jnp.array(y)
  return X, y

def unpack_hidden_dims_str(hidden_dims_str: str) -> list[int]:
  return [int(x) for x in hidden_dims_str.split(",")]


class LearnedOptimizerInMemoryImageDatasetTask(learned_optimizer_lib.Lopt_inner_task):

  def __init__(
    self,
    dataset_name: str,
    image_shape: Sequence[int], # a two-element tuple
    train_batch_size: int,
    val_batch_size: int,
    use_val: bool, # whether to use validation set (True) or training set (False) for outer loss computation
    use_accuracy_as_outer_loss: bool, # whether to use accuracy or cross entropy loss as outer loss
    inner_model_transformed_with_state: haiku.TransformedWithState, # the inner model that takes in batched X and returns batched logits ([batch_size, num_classes])
    inner_model_name: str,
    use_threshold_outer_cross_entropy_loss : bool = False,
    # whether to use thresholded cross entropy loss as outer loss
    ):
    self.dataset_name = dataset_name
    self.image_shape = image_shape

    # training set setup
    # the split [:80%] seems to be consistent
    self.train_Xs, self.train_ys = \
      get_entire_Xy_tuple_reshape(self.dataset_name, "train[0:80%]", self.image_shape)
    self.train_batch_size = train_batch_size

    # validation set setup
    self.use_val = use_val # whether to use validation set for outer loss computation
    if self.use_val:
      self.val_Xs, self.val_ys = \
        get_entire_Xy_tuple_reshape(self.dataset_name, "train[80%:]", self.image_shape)
    else:
      self.val_Xs, self.val_ys = self.train_Xs, self.train_ys
    self.val_batch_size = val_batch_size

    self.use_accuracy_as_outer_loss = use_accuracy_as_outer_loss
    self.use_threshold_outer_cross_entropy_loss = use_threshold_outer_cross_entropy_loss

    # inner model

    # self.net_without_state.init(rng, x)
    # self.net_without_state.apply(params, rng, x)
    # this might never be used
    # self.net_without_state = haiku.transform(_forward)
    # self.net_with_state.init(rng, x)
    # self.net_with_state.apply(params, state, rng, x)
    self.net_with_state = inner_model_transformed_with_state
    self.inner_model_name = inner_model_name

  def name(self,):
    s = (
      f"{self.dataset_name}{self.image_shape[0]}x{self.image_shape[1]}_"
      f"useval{self.use_val}_useacc{self.use_accuracy_as_outer_loss}_"
      f"tb{self.train_batch_size}_vb{self.val_batch_size}_"
      f"{self.inner_model_name}"
    )

    return s
  
  def inner_init_with_state(self, key) -> tuple[Params, State]:
    # here we need to make sure we are passing in a single example but
    # still with a batch dimension
    return self.net_with_state.init(key, self.train_Xs[:1])

  def inner_loss_with_state(self, params: Params, state: State, key: PRNGKey) -> tuple[jax.Array, State]:
    index_key, predict_key = jax.random.split(key)

    # sample a batch of training data to compute the inner loss for inner gradient
    sampled_indices = jax.random.choice(index_key, self.train_Xs.shape[0], (self.train_batch_size,), replace=False)
    sampled_train_Xs = jnp.take(self.train_Xs, sampled_indices, axis=0)
    sampled_train_ys = jnp.take(self.train_ys, sampled_indices, axis=0)
    # import ipdb; ipdb.set_trace()

    # compute the inner loss and the updated state
    sampled_train_logits, next_state = self.net_with_state.apply(params, state, predict_key, sampled_train_Xs)
    loss_per_example = optax.softmax_cross_entropy_with_integer_labels(sampled_train_logits, sampled_train_ys)
    mean_loss = jnp.mean(loss_per_example)

    return mean_loss, next_state

  def outer_loss_with_state(self, params: Params, state: State, key: PRNGKey) -> jax.Array:
    # here we don't update the inner model's state
    index_key, predict_key = jax.random.split(key)

    # sample a batch of validation data to compute the outer loss
    sampled_indices = jax.random.choice(index_key, self.val_Xs.shape[0], (self.val_batch_size,), replace=False)
    sampled_val_Xs = jnp.take(self.val_Xs, sampled_indices, axis=0)
    sampled_val_ys = jnp.take(self.val_ys, sampled_indices, axis=0)
    # import ipdb; ipdb.set_trace()

    # compute the outer loss
    sampled_val_logits, _ = self.net_with_state.apply(params, state, predict_key, sampled_val_Xs)
    if self.use_accuracy_as_outer_loss:
      predicted_ys = jnp.argmax(sampled_val_logits, axis=-1)
      accuracy = jnp.mean(jnp.equal(predicted_ys, sampled_val_ys))
      # we are minimize the negative accuracy
      outer_loss = -accuracy
    else:
      loss_per_example = optax.softmax_cross_entropy_with_integer_labels(sampled_val_logits, sampled_val_ys)
      if self.use_threshold_outer_cross_entropy_loss:
        # perform thresholding on the per example loss
        loss_per_example = jnp.clip(
          loss_per_example, a_min=None, a_max=1.5 * jnp.log(DATASET_TO_NUM_CLASSES[self.dataset_name.upper()]))
      outer_loss = jnp.mean(loss_per_example)
    
    return outer_loss

def create_inner_model_transformed_with_state() -> Tuple[haiku.TransformedWithState, str]:
  # return a tuple of (model, model_name)
  # where model is the haiku transformed model with state
  # here haiku requires the initialization of haiku module inside haiku.transform_with_state
  # function, this makes it impossible to access the .name() field of the model
  # thus we have to separately return the model name
  model_type = FLAGS.lopt_inner_model.upper()
  num_classes = DATASET_TO_NUM_CLASSES[FLAGS.lopt_dataset_name.upper()]
  if FLAGS.lopt_activation.upper() == "relu".upper():
    activation = jax.nn.relu
  elif FLAGS.lopt_activation.upper() == "gelu".upper():
    activation = jax.nn.gelu
  else:
    assert False, "unsupported activation"

  if model_type == "MLP":
    hidden_dims = unpack_hidden_dims_str(FLAGS.lopt_hidden_dims)
    dropout_rate = 0.0
    inner_model_transformed_with_state = haiku.transform_with_state(
      lambda x: learned_optimizer_inner_models.MLP(
      num_classes=num_classes,
      activation=activation,
      hidden_dims=hidden_dims,
      dropout_rate=dropout_rate,)(x)
    )
    inner_model_name = (
      f"MLP{'-'.join([str(x) for x in hidden_dims])}_"
      f"{activation.__name__}_"
      f"dropout{dropout_rate}"
    )

  elif model_type == "CNN":
    hidden_channels = unpack_hidden_dims_str(FLAGS.lopt_hidden_channels)
    inner_model_transformed_with_state =  haiku.transform_with_state(
      lambda x: learned_optimizer_inner_models.FullConvNet(
      num_classes=num_classes,
      activation=activation,
      hidden_channels=hidden_channels,)(x)
    )
    inner_model_name = (
      f"FullConvNet{'-'.join([str(x) for x in hidden_channels])}_"
      f"{activation.__name__}"
    )
  else:
    assert False, "unsupported inner model type"

  return inner_model_transformed_with_state, inner_model_name


def create_task_train_test():
  # return a train_task and test_task (they can be the same object)
  #################################
  ##### define the inner task #####
  #################################

  # activation function

  inner_model_transformed_with_state, inner_model_name = create_inner_model_transformed_with_state()
  print(inner_model_name)

  train_batch_size = FLAGS.lopt_batch_size
  val_batch_size = FLAGS.lopt_batch_size
  use_threshold_outer_cross_entropy_loss = FLAGS.use_threshold_outer_cross_entropy_loss
  # TESTED: verify this is the same for FashionMNIST training
  # TESTED: verify it's not the same if we explicitly set the flag for FashionMNIST
  # use_threshold_outer_cross_entropy_loss = False
  if FLAGS.lopt_dataset_name.upper() == "CIFAR10":
    assert FLAGS.use_threshold_outer_cross_entropy_loss

  inner_task = LearnedOptimizerInMemoryImageDatasetTask(
    dataset_name=FLAGS.lopt_dataset_name,
    image_shape=(FLAGS.image_length, FLAGS.image_length), # square image shape
    train_batch_size=train_batch_size,
    val_batch_size=val_batch_size,
    use_val=FLAGS.use_val,
    use_accuracy_as_outer_loss=FLAGS.use_accuracy,
    inner_model_transformed_with_state=inner_model_transformed_with_state,
    inner_model_name=inner_model_name,
    use_threshold_outer_cross_entropy_loss=use_threshold_outer_cross_entropy_loss,
  )

  ##########################################
  ##### define the lopt dynamics model #####
  ##########################################

  ##### architecture of meta-learned optimizer #####
  if FLAGS.lopt_architecture.upper() == "LearnableSGDM".upper():
    lopt = lopt_base.LearnableSGDM()
  elif FLAGS.lopt_architecture.upper() == "LearnableAdam".upper():
    lopt = lopt_base.LearnableAdam()
  elif FLAGS.lopt_architecture.upper() == "MLPLOpt".upper():
    kwargs = {}
    if FLAGS.lopt_mlplopt_hidden_size is not None:
      kwargs["hidden_size"] = FLAGS.lopt_mlplopt_hidden_size
    if FLAGS.lopt_mlplopt_num_hidden_layers is not None:
      kwargs["hidden_layers"] = FLAGS.lopt_mlplopt_num_hidden_layers
    lopt = mlp_lopt.MLPLOpt(**kwargs)
  else:
    raise ValueError("not supported optimizer architecture")

  theta_init = None

  lopt_dynamics = learned_optimizer_lib.LearnedOptimizerDynamics(
    inner_task=inner_task,
    T=FLAGS.horizon_length,
    learned_opt=lopt,
    single_sequence_seed=FLAGS.lopt_single_sequence_seed,
    theta_init=theta_init,)
  
  return lopt_dynamics, lopt_dynamics

def create_truncated_step_train_test(train_task, test_task, num_test_particles=None):
  # return a tuple of (truncated_step_train, truncated_step_test)

  # only use random initial iteration if using truncation methods
  # if using full trajectory, always start from the beginning (not using random initial iteration)
  random_initial_iteration_train = \
    gradient_estimator_utils.use_random_initial_iteration_for_truncated_step_train()
  
  ### the version below might cause a RET_CHECK error so we use non mutli device
  ### version below.
  # truncated_step_train = \
  #   dynamical_system_truncated_step.DynamicalSystemDecomposableTruncatedStepMultiDevice(
  #     dynamical_system=train_task,
  #     num_tasks=(FLAGS.num_particles // FLAGS.num_gradient_estimators),
  #     T=FLAGS.horizon_length,
  #     random_initial_iteration=random_initial_iteration_train,
  #     truncation_window_size=FLAGS.trunc_length,
  #     # for full trajectory methods like FullES and FullGradient
  #     # the random_initial_iteration will be turned off so even though
  #     # truncation_window_size is set to arbitrary values it doesn't matter
  #     device_list=jax.devices(),
  #   )

  ##### we use the non multi device to avoid a RET_CHECK issue
  truncated_step_train = \
    dynamical_system_truncated_step.DynamicalSystemDecomposableTruncatedStep(
      dynamical_system=train_task,
      num_tasks=(FLAGS.num_particles // FLAGS.num_gradient_estimators),
      T=FLAGS.horizon_length,
      random_initial_iteration=random_initial_iteration_train,
      truncation_window_size=FLAGS.trunc_length,
      # for full trajectory methods like FullES and FullGradient
      # the random_initial_iteration will be turned off so even though
      # truncation_window_size is set to arbitrary values it doesn't matter
    )
  
  if num_test_particles is None:
    if FLAGS.lopt_single_sequence_seed is not None:
      # if we use a single sequence seed, we only need to test on one particle
      num_test_particles = 1
    else:
      # the default when num_test_particles is not specified
      num_test_particles = 100

  truncated_step_test = \
    dynamical_system_truncated_step.DynamicalSystemDecomposableTruncatedStep(
      dynamical_system=test_task,
      num_tasks=num_test_particles, # probably need to set an appropriate number depending on the variance of the dataset
      T=FLAGS.horizon_length,
      random_initial_iteration=False,
    )

  return (truncated_step_train, truncated_step_test)

def create_truncated_step_train_for_evaluation(train_task, num_particles=100):

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