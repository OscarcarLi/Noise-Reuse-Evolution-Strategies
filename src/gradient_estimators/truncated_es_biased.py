"""TruncatedES (biased estimator, introduced in https://arxiv.org/abs/1810.10180)"""
import functools
from typing import Mapping, Sequence, Tuple, Any

import haiku as hk
import jax
import jax.numpy as jnp
from src.utils import profile
from src.utils import summary
from src.utils import tree_utils
from src.utils import common
from src.outer_trainers import gradient_learner
from src.task_parallelization import truncated_step
import chex


PRNGKey = jax.Array
MetaParams = Any
UnrollState = Any
TruncatedUnrollState = Any

import flax.struct

# this is different from vector_sample_perturbations
# as we don't need the positive and negatively perturbed thetas
# (theta, key, std) # only keys are different over multiple samples
sample_multiple_perturbations = jax.jit(jax.vmap(common.sample_perturbations, in_axes=(None, 0, None)))

@flax.struct.dataclass
class TruncatedESBiasedSingleStateAttributes(gradient_learner.GradientEstimatorState):
  current_unroll_states: TruncatedUnrollState

@flax.struct.dataclass
class TruncatedESBiasedAntitheticAttributes:
  pos_unroll_states: TruncatedUnrollState
  neg_unroll_states: TruncatedUnrollState


class TruncatedESBiased(gradient_learner.GradientEstimator):
  """TruncatedES, first introduced in 
  Understanding and correcting pathologies in the training of learned optimizers
  https://arxiv.org/abs/1810.10180,
  Formalized in the PES paper.
  """

  def __init__(
      self,
      truncated_step: truncated_step.TruncatedStep,
      unroll_length: int = 20,
      std: float = 0.01,
      burn_in_length: int = 0,
      jitted: bool = True,
  ):
    """Initializer.

    Args:
      truncated_step: class containing functions for initializing and
        progressing a inner-training state.
      unroll_length: length of the unroll truncation window.
      std: standard deviation for ES.
      burn_in_length: how many steps to run unroll starting from the inner states
        returned from truncated_step.init_step_state to ensure all the states are
        obtained by using the current theta (+ epsilon)
      jitted: boolean whether to jit the core computations
    """
    self.truncated_step = truncated_step
    self.unroll_length = unroll_length
    self.std = std
    self.burn_in_length = burn_in_length
    self._total_env_steps_used = 0
    self.jitted = jitted
    if self.jitted:
      self.single_state_unroll = jax.jit(self._single_state_unroll)
      self.antithetic_es_unroll = jax.jit(self._antithetic_es_unroll)
      self.prepare = jax.jit(self._prepare, static_argnums=(3,))
    else:
      self.single_state_unroll = self._single_state_unroll
      self.antithetic_es_unroll = self._antithetic_es_unroll
      self.prepare = self._prepare

  @staticmethod
  def _prepare(key, theta, std, num_tasks, state):
    eps_key, unroll_key = jax.random.split(key, num=2)

    epsilon_key_list = jax.random.split(eps_key, num_tasks)
    # we use sample_perturbations instead of vector_sample_perturbations
    # as we don't need the positively/negatively perturbed thetas
    # one epsilon vector for each particle
    epsilons = sample_multiple_perturbations(
      theta, epsilon_key_list, std)

    # this would not work with mutable current_unroll_states (as implemented in RL open ai gym)
    # thus we implement a deepcopy immutable version through openai_gym_truncated_step
    # we create the unroll_keys here so that we can use the same key sequence for
    # both single state and antithetic state unrolling
    unroll_key_list = jax.random.split(unroll_key, num_tasks)
    antithetic_state = TruncatedESBiasedAntitheticAttributes(
      pos_unroll_states=state.current_unroll_states,
      neg_unroll_states=state.current_unroll_states,)
    return epsilons, unroll_key_list, antithetic_state

  def grad_est_name(self):
    return ("TruncatedESBiased_"
            f"N={self.truncated_step.num_tasks},K=W={self.unroll_length},sigma={self.std}")

  def task_name(self):
    return self.truncated_step.task_name()

  @property
  def total_env_steps_used(self,):
    return self._total_env_steps_used

  @profile.wrap()
  def init_worker_state(self, worker_weights: gradient_learner.WorkerWeights,
                        key: PRNGKey) -> TruncatedESBiasedSingleStateAttributes:
    key1, key2 = jax.random.split(key, num=2)
    current_unroll_states = self.truncated_step.init_step_state(
        worker_weights.theta,
        worker_weights.outer_state,
        jax.random.split(key1, self.truncated_step.num_tasks),
        theta_is_vector=False)
    truncated_es_biased_single_state_attributes = TruncatedESBiasedSingleStateAttributes(
      current_unroll_states=current_unroll_states,)
    
    # a burn-in period to ensure the theta's have its states all unrolled by itself
    if self.burn_in_length > 0:
      rng = hk.PRNGSequence(key2)  # type: ignore
      for _ in range(self.burn_in_length):
        data = self.truncated_step.get_batch()
        curr_key = next(rng)
        truncated_es_biased_single_state_attributes, _, _ =\
          self.single_state_unroll(
            worker_weights.theta,
            truncated_es_biased_single_state_attributes,
            curr_key,
            data,
            worker_weights.outer_state)
      self._total_env_steps_used += int(jnp.sum(truncated_es_biased_single_state_attributes.current_unroll_states.inner_step))
    
    return truncated_es_biased_single_state_attributes

  @profile.wrap()
  def compute_gradient_estimate(
      self,
      worker_weights,
      key: PRNGKey,
      state: TruncatedESBiasedSingleStateAttributes, # this is the same state returned by init_worker_state
      with_summary=False,
  ) -> Tuple[gradient_learner.GradientEstimatorOut, Mapping[str, jax.Array]]:
    theta = worker_weights.theta
    outer_state = worker_weights.outer_state

    # get the entire sequence of data (unroll_length steps and each step has num_tasks data
    # one for each particle)
    data_sequence = self.truncated_step.get_batch(steps=self.unroll_length)

    # the prepare function is jitted to make computation faster (defined above)
    epsilons, unroll_key_list, antithetic_state = \
      self.prepare(key, theta, self.std, self.truncated_step.num_tasks, state)
    # import ipdb; ipdb.set_trace()

    ##### compute the gradient estimates #####
    loss_sum = jnp.array(0.0)
    g_sum = jax.tree_util.tree_map(jnp.zeros_like, theta)
    # cannot use jnp.zeros(shape=1) for total_count (this will be a length 1 array which is undesired)
    total_antithetic_count = jnp.array(0)
    for i in range(self.unroll_length):
      antithetic_state, loss_sum_step, g_sum_step, count = \
        self.antithetic_es_unroll(
        theta=theta, # a single theta
        epsilons=epsilons, # one epsilon for each particle
        state=antithetic_state,
        key=jax.tree_util.tree_map(lambda x: x[i], unroll_key_list),
        data=jax.tree_util.tree_map(lambda x: x[i], data_sequence),
        outer_state=outer_state)
      loss_sum += loss_sum_step
      g_sum = jax.tree_util.tree_map(lambda x, y: x + y, g_sum, g_sum_step)
      total_antithetic_count += count
      self._total_env_steps_used += 2 * self.truncated_step.num_tasks

    g = jax.tree_util.tree_map(lambda x: x / total_antithetic_count, g_sum)

    ##### actually unroll the system #####
    # loss_sum = 0.
    # total_single_count = 0.
    for i in range(self.unroll_length):
      state, loss_sum_step, count = self.single_state_unroll(
        theta=theta,
        state=state,
        key=jax.tree_util.tree_map(lambda x: x[i], unroll_key_list),
        data=jax.tree_util.tree_map(lambda x: x[i], data_sequence),
        outer_state=outer_state)
      self._total_env_steps_used += self.truncated_step.num_tasks
      del loss_sum_step
      del count
      # loss_sum += loss_sum_step
      # total_single_count += count
    
    output = gradient_learner.GradientEstimatorOut(
        mean_loss=loss_sum / (2 * total_antithetic_count),
        grad=g,
        unroll_state=state,
        unroll_info=None,
    )
    return output, {}

  def _single_state_unroll(
    self,
    theta, # a single theta
    state: TruncatedESBiasedSingleStateAttributes,
    key,
    data,
    outer_state):
    current_unroll_states = state.current_unroll_states
    next_unroll_states, outs = self.truncated_step.unroll_step(
      theta=theta,
      unroll_state=current_unroll_states,
      key_list=jax.random.split(key, self.truncated_step.num_tasks),
      data=data,
      outer_state=outer_state,
      theta_is_vector=False)
    return \
      (
        TruncatedESBiasedSingleStateAttributes(
          current_unroll_states=next_unroll_states),
      jnp.sum(outs.loss * outs.mask), # sum of loss
      jnp.sum(outs.mask) # count
      )

  def _antithetic_es_unroll(
    self,
    theta: MetaParams, # a single theta
    epsilons: jax.Array, # one epsilon for each particle
    state: TruncatedESBiasedAntitheticAttributes, 
    key: chex.PRNGKey,
    data: Any, # single batch of data to be used for both pos and neg unroll
    outer_state: Any) -> Tuple[TruncatedESBiasedAntitheticAttributes, jax.Array, MetaParams, jax.Array]:

    # unpack the current state
    pos_unroll_states, neg_unroll_states = \
      state.pos_unroll_states, state.neg_unroll_states
    
    # compute the perturbation for this unroll step
    # there is one perturbed pos/neg theta for each trajectory
    pos_perturbed_thetas = jax.tree_util.tree_map(lambda a,b: jnp.expand_dims(a, 0) + b, theta, epsilons)
    neg_perturbed_thetas = jax.tree_util.tree_map(lambda a,b: jnp.expand_dims(a, 0) - b, theta, epsilons)

    pos_unroll_states, pos_outs = \
      self.truncated_step.unroll_step(
        theta=pos_perturbed_thetas,
        unroll_state=pos_unroll_states,
        key_list=jax.random.split(key, self.truncated_step.num_tasks),
        data=data,
        outer_state=outer_state,
        theta_is_vector=True)
    neg_unroll_states, neg_outs = \
      self.truncated_step.unroll_step(
        theta=neg_perturbed_thetas,
        unroll_state=neg_unroll_states,
        key_list=jax.random.split(key, self.truncated_step.num_tasks),
        # using the same key as pos ensures when resetting we get the same initial inner state and
        # and also ensures we get the same loss evaluation (if it takes randomness)
        data=data, # also use the same data
        outer_state=outer_state,
        theta_is_vector=True)
    
    # pos_outs.loss is an array of losses (one for each trajectory/particle)
    # need to exclude the 0 entries when the trajectory has resetted and a loss of 0 is returned. 
    # we set the multiplicative weight for the particle that resets to 0
    multiplier = ((pos_outs.loss - neg_outs.loss) * pos_outs.mask) * 1 / (2 * self.std ** 2)
    loss_sum_step = jnp.sum(pos_outs.loss * pos_outs.mask + neg_outs.loss * neg_outs.mask)
    # g_sum_step is equal to the sum of gradient estimates for all particles
    # which has non trivial gradient estimates (this excludes the particles)
    # that has reset in this unroll.                            
    g_sum_step = jax.tree_util.tree_map(lambda eps:
            jnp.sum( # sum over all particles (each particle is working on one trajectory)
              eps * jnp.reshape(multiplier, [multiplier.shape[0]] + [1]*(len(eps.shape)-1)),
              axis=0),
                              epsilons)
    # count keeps track of how many gradient estimates are summed in this unroll
    count = jnp.sum(pos_outs.mask)

    return (
      TruncatedESBiasedAntitheticAttributes(
              pos_unroll_states=pos_unroll_states,
              neg_unroll_states=neg_unroll_states,),
      loss_sum_step,
      g_sum_step,
      count)
