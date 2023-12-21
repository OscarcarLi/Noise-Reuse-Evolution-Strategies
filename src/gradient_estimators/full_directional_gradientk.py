# coding=utf-8
"""full trajectory forward mode directional gradient descent.
"""
import functools
from typing import Mapping, Sequence, Tuple, Any

import haiku as hk
import jax
import jax.numpy as jnp
from src.utils import profile
from src.utils import summary
from src.utils import tree_utils
from src.utils import common
from src.task_parallelization import truncated_step

from src.gradient_estimators import truncated_forwardmodek

import chex


PRNGKey = jax.Array
MetaParams = Any
UnrollState = Any
TruncatedUnrollState = Any

import flax
import ipdb

# this is different from vector_sample_perturbations
# as we don't need the positive and negatively perturbed thetas
# (theta, key, std) # only keys are different over multiple samples
sample_multiple_perturbations = jax.jit(jax.vmap(common.sample_perturbations, in_axes=(None, 0, None)))

class FullDirectionalGradientK(truncated_forwardmodek.TruncatedForwardModeK):
  """Directional gradient descent estimator that runs full episodes,
  sample a new noise every K steps"""

  def __init__(
      self,
      truncated_step: truncated_step.TruncatedStep,
      K: int,
      T: int,
  ):
    """Initializer.

    Args:
      truncated_step: class containing functions for initializing and
        progressing a inner-training state.
      K: how many unrolls to sample a new epsilon
      T: horizon length
    """
    super(FullDirectionalGradientK, self).__init__(
      truncated_step=truncated_step,
      unroll_length=T,
      K=K,
      burn_in_length=0,) # here we need to make sure there is no random init in truncated_step

  def grad_est_name(self):
    # there is no sigma parameter for TruncatedForwardMode
    return \
      ("FullDirectionalGradientK"
      f"_N={self.truncated_step.num_tasks},K={self.K},W=T")


class FullDirectionalGradient(FullDirectionalGradientK):
  """Directional gradient descent estimator that runs full episodes,
  sample a single noise for each particle's episode"""
  def __init__(
      self,
      truncated_step: truncated_step.TruncatedStep,
      T: int,
  ):
    """Initializer.

    Args:
      truncated_step: class containing functions for initializing and
        progressing a inner-training state.
      T: horizon length
    """
    super(FullDirectionalGradient, self).__init__(
      truncated_step=truncated_step,
      K=T,
      T=T,
    )
  
  def grad_est_name(self):
    # there is no sigma parameter for TruncatedForwardMode
    return \
      ("FullDirectionalGradient"
      f"_N={self.truncated_step.num_tasks},W=T")
