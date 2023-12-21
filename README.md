# Variance-Reduced Gradient Estimation via Noise Reuse in Online Evolution Strategies

### The code repository contains the code and experiments for the paper.

> [Variance-Reduced Gradient Estimation via Noise Reuse in Online Evolution Strategies](https://arxiv.org/abs/2304.12180)
>
> Unrolled computation graphs are prevalent throughout machine learning but present challenges to automatic differentiation (AD) gradient estimation methods when their loss functions exhibit extreme local sensitivtiy, discontinuity, or blackbox characteristics. In such scenarios, online evolution strategies methods are a more capable alternative, while being more parallelizable than vanilla evolution strategies (ES) by interleaving partial unrolls and gradient updates. In this work, we propose a general class of unbiased online evolution strategies methods. We analytically and empirically characterize the variance of this class of gradient estimators and identify the one with the least variance, which we term Noise-Reuse Evolution Strategies (NRES). Experimentally, we show NRES results in faster convergence than existing AD and ES methods in terms of wall-clock time and number of unroll steps across a variety of applications, including learning dynamical systems, meta-training learned optimizers, and reinforcement learning.

## Dependencies
To have the correct dependencies, we suggest creating a conda virtual environment and installing the required packages by following the step-by-step installation instructions in `DEPENDENCIES.md`. For completeness, we also include the complete dependencies of our virtual environment in `environment.yml`. However, this specific set of dependencies might not directly work on your system+accelerator combination so we suggest first following the list of steps given in `DEPENDENCIES.md`.

## Understanding the codebase structure
The entry point of our training code is in the file `main_training.py`. The command line flags to configure the training is defined using `absl.flags` in `main_training.py` (general configuration of meta-training, gradient estimator, and eval) and in `lorenz_utils.py`, `lopt_utils.py`, `rl_utils.py` (application specific configuration). The meaning of each flag is given where the flag is defined.

For GPU based computation (Lorenz system and learned optimizer), we abstract the interface of the unrolled computation graph through an abc class `src.task.dynamical_system.DynamicalSystemDecomposable` that supports the initialization, unrolling, loss calculation, and unroll resetting (after reaching the end of the current episode after T unrolls) of a single trajectory.

To enable parallelization over multiple trajectories, we then provide a wrapper abstract class `src.task_parallelization.truncated_step.TruncatedStep` which can 1) initialize a vector of inner states (possibly step-unlocked) and 2) given a shared `theta` or a vector of `theta`'s (one for each episode trajectory), unroll each inner state under the corresponding `theta` for one step and output a new vector of inner states. (Notice jax requires functional programming style so the class `TruncatedStep` doesn't keep the inner states as attributes.)

For the Mujoco reinforcement learning task, because the environment transition is computed over CPUs, we are unable to perform the `jax.vmap` parallelization over multiple indendent episodes in Jax. As a result, we provide a customized subclass implementation of `TruncatedStep` in `src.task_parallelization.openai_gym_truncated_step.OpenAIGymTruncatedStep` which manages multiple environments for init and unrolls.

To compute the gradients, we use the `TruncatedStep` object to create a gradient estimator (abstract class `src.outer_trainers.gradient_learner.GradientEstimator`). This estimator allows us to initialize stateful gradients estimators and use these estimators states to compute the gradient to be used in any meta-level first-order optimizer.

## Reproducing the experiments
We provide the training commands to reproduce all the three experiment applications presented in the main paper. 

To run the experiments on the Lorenz system parameter learning application, follow the commands (line by line for each gradient estimator) in the bash script
`scripts/dynamical_system_lorenz/lorenz_script.sh`.

To run the experiments on meta-training learned optimizer, follow the commands in the bash script `scripts/lopt/lopt_fashion_mnist_mlp_script.sh` and `scripts/lopt/lopt_cifar10_cnn_script.sh`.

To run the experiments for the reinforcement learning mujoco tasks, follow the commands (line by line) in the bash script `scripts/rl/rl_swimmer_script.sh` and `scripts/rl/rl_halfcheetah_script.sh`.

The theta trajectory and tensorboard event files will be saved at a log folder in `./runs` and can be used to generate the plots in the paper. To visualize the learning progress, we use tensorboard. As an example, to visualize the loss progression for the learned optimizer task on fashion mnist, use the following command:

```python
tensorboard --logdir ./runs/lopt/fashion_mnist8x8_usevalTrue_useaccFalse_tb128_vb128_MLP32-32-32_gelu_dropout0.0_singleseq88_T1000_MLPLOpt --load_fast=true
```

## Metrics of interest
_Objectives_: For the Lorenz dynamical system learning and meta-training learned optimizers, the performance metric (the MSE generalization loss for Lorenz, meta-training loss for learned optimizer) is recorded under the tag `test/unperturbed_meta_test_loss` in tensorboard. For the Mujoco RL task, the reward progression is recorded under the tag `test/unperturbed_test_reward` in tensorboard.

_Wall clock time_: Because tensorboard logging is time stamped, we can retrieve the time information from the events files. Here we have taken care to not compute any evaluation during the training phase to ensure the timing measurements' correctness. For all the timed experiments, the evaluation is performed only after the training has finished using the flag `--run_test_after_training`.

_Total number of unroll steps_: We track this number under the tab `total_env_steps` in tensorboard. For the step-unlocked online gradient estimation methods, we have already accounted for the wasted unroll steps used to initialize the step-unlocked workers.

_Number of sequential unroll steps_: When calculating the number of sequential unroll steps, for each $\theta$-update iteration, the offline method FullES uses $T$ sequential steps, while the online ES methods (PES, TES, NRES) use $W$ steps. (This assumes all the workers run in parallel).

## Comments, Questions, and Issues
Thanks a lot for your interest in our work! For questions and issues regarding the code, please file issues on github or email at <oscarli@cmu.edu>.

## Citing the work
```
@inproceedings{li2023variance,
  title={Variance-Reduced Gradient Estimation via Noise-Reuse in Online Evolution Strategies},
  author={Li, Oscar and Harrison, James and Sohl-Dickstein, Jascha and Smith, Virginia and Metz, Luke},
  booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
  year={2023}
}
```