# coding=utf-8
# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Common functions for outer trainers."""

import functools
from typing import Any, Mapping, Optional, Tuple, TypeVar

import chex
import jax
import jax.numpy as jnp
from flax.training import checkpoints

from src.utils import summary
from src.utils import tree_utils

MetaParams = Any
OuterState = Any
UnrollState = Any

T = TypeVar("T")
G = TypeVar("G")

def load_theta(theta_template, step, saved_path):
  theta = checkpoints.restore_checkpoint(
      ckpt_dir=saved_path,
      target=theta_template,
      step=step,
      prefix="theta",)
  return theta 

def load_theta_trajectory(theta_template, saved_path):
  """load theta trajectory from saved_path
  theta_template is the pytree template of the theta
  
  It is assumed that theta trajectory is saved under name
  theta_trajectory0
  """
  theta = checkpoints.restore_checkpoint(
      ckpt_dir=saved_path,
      target=theta_template,
      step=0,
      prefix="theta_trajectory",)
  return theta

@jax.jit
def sample_perturbations(variables: T, rng: chex.PRNGKey, std: float) -> T:
  flat, tree_def = jax.tree_util.tree_flatten(variables)
  rngs = jax.random.split(rng, len(flat))
  perturbs = []
  for key, f in zip(rngs, flat):
    perturbs.append(jax.random.normal(key, shape=f.shape, dtype=f.dtype) * std)
  return jax.tree_util.tree_unflatten(tree_def, perturbs)


@functools.partial(jax.jit, static_argnums=(3,))
def vector_sample_perturbations(theta: T, key: chex.PRNGKey, std: float,
                                num_samples: int) -> Tuple[T, T, T]:
  """Sample multiple antithetic ES perturbations."""

  def _fn(key):
    pos = sample_perturbations(theta, key, std=std)
    p_theta = jax.tree_util.tree_map(lambda t, a: t + a, theta, pos)
    n_theta = jax.tree_util.tree_map(lambda t, a: t - a, theta, pos)
    return pos, p_theta, n_theta

  keys = jax.random.split(key, num_samples)
  vec_pos, vec_p_theta, vec_n_theta = jax.vmap(_fn)(keys)
  return vec_pos, vec_p_theta, vec_n_theta


# TODO(lmetz) buffer dontation here, and in the next function is not
# taken use of by XLA meaning 1 additional copy is happening.
@functools.partial(jax.jit, donate_argnums=(0,), static_argnames=("axis",))
def _split_tree(tree, axis=0):
  """Split the provided tree in half along `axis`."""
  assert axis in [0, 1]
  if axis == 0:
    num_tasks = tree_utils.first_dim(tree) // 2
    a = jax.tree_map(lambda x: x[0:num_tasks], tree)
    b = jax.tree_map(lambda x: x[num_tasks:], tree)
    return a, b
  elif axis == 1:
    num_tasks = jax.tree_leaves(tree)[0].shape[1] // 2
    a = jax.tree_map(lambda x: x[:, 0:num_tasks], tree)
    b = jax.tree_map(lambda x: x[:, num_tasks:], tree)
    return a, b


@functools.partial(jax.jit, donate_argnums=(0, 1), static_argnames=("axis",))
def _stack(a, b, axis=0):
  return jax.tree_map(lambda x, y: jnp.concatenate([x, y], axis=axis), a, b)