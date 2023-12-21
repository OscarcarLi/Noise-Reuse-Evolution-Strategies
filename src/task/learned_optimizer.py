"""dynamical system implementation of learned optimizer"""
import abc
from typing import Any, Sequence, Optional, Tuple

import jax
import jax.numpy as jnp
import haiku
import flax.struct

from src.task import dynamical_system
from src.learned_optimizers import base as lopt_base
from src.learned_optimizers import mlp_lopt

InnerState = Any  # inner state
PRNGKey = Any
Params = Any
# this is the inner model's state for prediction (somethings like exponential moving average, batch stats)
State = Any
MetaParams = Any

class Lopt_inner_task(abc.ABC):
  @abc.abstractmethod
  def name(self):
    """return the string description of the task"""
  
  @abc.abstractmethod
  def inner_init_with_state(self, key) -> tuple[Params, State]:
    """return the initial parameters and state of the task"""
  
  @abc.abstractmethod
  def inner_loss_with_state(
        self,
        params: Params,
        state: State,
        key: PRNGKey) -> tuple[jax.Array, State]:
    """return the inner loss and the update state of the task to compute the inner gradient of
    the key will handle random inner batch selection and also forward randomness"""
    # when we want to use batchnorm and have the inner model state to
    # be updated after the inference (with inner loss calculation)
    # we need to pass in is_training=True
  
  @abc.abstractmethod
  def outer_loss_with_state(
        self,
        params: Params,
        state: State,
        key: PRNGKey) -> jax.Array:
    """return the outer loss of the task after an inner step
    here the outer_loss can be non-differentiable metrics like accuracy
    here the state is not updated"""
    # for model with batch norm, to not have outer loss update
    # the batch stats, we should pass in is_training=False in apply function



@flax.struct.dataclass
class LearnedOptimizerInnerState:
  learned_opt_state: Any
  inner_step: jax.Array

class LearnedOptimizerDynamics(dynamical_system.DynamicalSystemDecomposable):
  def __init__(
    self,
    inner_task: Lopt_inner_task,
    T: int,
    learned_opt: lopt_base.LearnedOptimizer,
    single_sequence_seed: Optional[int] = None,
    theta_init: MetaParams = None,
  ):
    self.inner_task = inner_task
    self.T = T
    self.learned_opt = learned_opt
    
    self.single_sequence_seed = single_sequence_seed
    self.use_single_sequence = self.single_sequence_seed is not None

    # this is to use .get_params_state() of the induced optimizer
    self.opt_template = \
      self.learned_opt.opt_fn(self.learned_opt.init(jax.random.PRNGKey(0)))

    if self.use_single_sequence:
      self.inner_init_key, self.unroll_key_root = \
        jax.random.split(jax.random.PRNGKey(self.single_sequence_seed), 2) # type: ignore
      self.unroll_key_list = jax.random.split(self.unroll_key_root, self.T)
    self.theta_init = theta_init

  def name(self,):
    inner_task_name = f"{self.inner_task.name()}_"
    if self.use_single_sequence:
      inner_task_name = inner_task_name + f"singleseq{self.single_sequence_seed}_"
    
    return \
      inner_task_name + (
      f"T{self.T}_"
      f"{self.learned_opt.name()}"
    )

  def meta_init(self, key: PRNGKey) -> MetaParams:
    if self.theta_init is not None:
      return self.theta_init
    else:
      return self.learned_opt.init(key)

  def inner_init(self, theta: MetaParams, key: PRNGKey) -> InnerState:
    if self.use_single_sequence:
      key = self.inner_init_key

    model_init_key, lopt_init_key = jax.random.split(key)
    inner_model_param, inner_model_state = \
      self.inner_task.inner_init_with_state(model_init_key)
    # opt_fn accepts an additional is_training flag but currently it's never used
    opt_state = self.learned_opt.opt_fn(theta).init(
      inner_model_param, inner_model_state,
      num_steps=self.T, key=lopt_init_key)
    
    return LearnedOptimizerInnerState(
      learned_opt_state=opt_state,
      inner_step=jnp.asarray(0, dtype=jnp.int32),
    )

  def _single_seq_unroll_key_split(self, inner_step) -> tuple[PRNGKey, PRNGKey, PRNGKey]:
    # use the same key for the same inner_step
    unroll_key = self.unroll_key_list[inner_step]
    unroll_without_reset_key, loss_evaluation_key, reset_key = jax.random.split(unroll_key, 3)
    return unroll_without_reset_key, loss_evaluation_key, reset_key

  def _single_seq_unroll_without_reset_key(self, state) -> PRNGKey:
      return self._single_seq_unroll_key_split(state.inner_step)[0]

  def _single_seq_loss_evaluation_key(self, state) -> PRNGKey:
    # we subtract 1 because the currently state.inner_step should be between [1, T] (both inclusive)
    return self._single_seq_unroll_key_split(state.inner_step - 1)[1]

  def _single_seq_state_reset_key(self, state) -> PRNGKey:
    # we subtract 1 because the currently state.inner_step should be between [1, T] (both inclusive)
    return self._single_seq_unroll_key_split(state.inner_step - 1)[2]
    

  def unroll_without_reset(self, state: InnerState, theta: MetaParams, data, key: PRNGKey) -> InnerState:
    if self.use_single_sequence:
      # use the same key for the same inner_step
      unroll_without_reset_key = self._single_seq_unroll_without_reset_key(state)
    else:
      unroll_without_reset_key = key

    inner_loss_key, opt_update_key = jax.random.split(unroll_without_reset_key)
    opt = self.learned_opt.opt_fn(theta) # is_training flag is not used
    learned_opt_state = state.learned_opt_state
    p, s = opt.get_params_state(learned_opt_state)
    (inner_loss, s), g = \
      jax.value_and_grad(self.inner_task.inner_loss_with_state, has_aux=True)(
        p, s, inner_loss_key,)
    next_learned_opt_state = opt.update(
      opt_state=learned_opt_state,
      grad=g,
      loss=inner_loss,
      model_state=s,
      key=opt_update_key,
    )
    return \
      LearnedOptimizerInnerState(
        learned_opt_state=next_learned_opt_state,
        inner_step=state.inner_step + 1,)

  def loss_evaluation(self, state: InnerState, data, key: PRNGKey) -> jax.Array:
    if self.use_single_sequence:
      loss_evaluation_key = self._single_seq_loss_evaluation_key(state)
    else:
      loss_evaluation_key = key
    learned_opt_state = state.learned_opt_state
    p, s = self.opt_template.get_params_state(learned_opt_state)
    outer_loss = self.inner_task.outer_loss_with_state(p, s, loss_evaluation_key)
    return outer_loss

  def state_reset_if_necessary(self, state: InnerState, theta: MetaParams, key: PRNGKey) -> Tuple[bool, InnerState]:
    is_done = state.inner_step >= self.T
    def true_fn():
      # this trajectory is finished, reset to a new start
      reset_state = self.inner_init(theta, key)
      return reset_state
    def false_fn():
      return state
    next_state = jax.lax.cond(is_done, true_fn, false_fn)
    return is_done, next_state


if __name__ == "__main__":
  """python3 -m src.task.learned_optimizer"""
  import numpy as np
  import tensorflow as tf
  tf.config.experimental.set_visible_devices([], "GPU")
  from src.task import learned_optimizer_inner_models
  import src.learned_optimizers.base as lopt_base
  from src.applications import lopt_utils

  ##### checking the function get_entire_Xy_tuple_reshape
  # X, y = lopt_utils.get_entire_Xy_tuple_reshape(
  #   dataset_name="mnist",
  #   split="train[:80%]",
  #   image_shape=(8, 8),
  # )
  # import ipdb; ipdb.set_trace()

  #####
  # use_accuracy = True
  use_accuracy = False

  # lopt = mlp_lopt.MLPLOpt()

  ################# MLP #################
  # T = 1000
  # lopt = lopt_base.LearnableSGD(initial_lr=0.03)
  # dataset_name = "fashion_mnist"
  # image_length = 8
  # num_classes = 10
  # activation = jax.nn.gelu
  # hidden_dims = [32, 32, 32]
  # dropout_rate = 0.0
  # inner_model_transformed_with_state = haiku.transform_with_state(
  #   lambda x: learned_optimizer_inner_models.MLP(
  #   num_classes=num_classes,
  #   activation=activation,
  #   hidden_dims=hidden_dims,
  #   dropout_rate=dropout_rate,)(x)
  # )
  # inner_model_name = (
  #   f"MLP{'-'.join([str(x) for x in hidden_dims])}_"
  #   f"{activation.__name__}_"
  #   f"dropout{dropout_rate}"
  # )
  ################# end of MLP #################

  ################# FullConvNet #################
  T = 1000
  lopt = lopt_base.LearnableAdam(initial_lr=0.001)
  dataset_name = "cifar10"
  image_length = 32
  num_classes = 10
  activation = jax.nn.relu
  hidden_channels = [32, 32, 32, 32]
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
  ################# end of FullConvNet #################

  inner_task = lopt_utils.LearnedOptimizerInMemoryImageDatasetTask(
    dataset_name=dataset_name,
    image_shape=(image_length, image_length), # square image shape
    train_batch_size=128,
    val_batch_size=128,
    use_val=True,
    use_accuracy_as_outer_loss=use_accuracy,
    inner_model_transformed_with_state=inner_model_transformed_with_state,
    inner_model_name=inner_model_name,
  )

  lopt_dynamics = LearnedOptimizerDynamics(
    inner_task=inner_task,
    T=T + 1,
    learned_opt=lopt,)

  rng = haiku.PRNGSequence(42)
  theta = lopt_dynamics.meta_init(key=jax.random.PRNGKey(15))
  inner_state = lopt_dynamics.inner_init(theta, key=next(rng))
  out_list = []
  outerloss_list = []

  unroll_function = jax.jit(lopt_dynamics.unroll)
  # unroll_function = lopt_dynamics.unroll

  for i in range(T):
    print(i)
    out = unroll_function(inner_state, theta, None, next(rng))
    inner_state = out.inner_state
    # import ipdb; ipdb.set_trace()
    # out_list.append(out)
    outerloss_list.append(float(out.loss))
  
  import ipdb; ipdb.set_trace()