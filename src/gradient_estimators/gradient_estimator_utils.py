from absl import flags
import types # for overriding methods in an already defined class
from src.gradient_estimators import truncated_es_biased
from src.gradient_estimators import truncated_nres
from src.gradient_estimators import truncated_pesk
from src.gradient_estimators import truncated_forwardmode
from src.gradient_estimators import truncated_forwardmodek
from src.gradient_estimators import uoro
from src.gradient_estimators import full_directional_gradientk
from src.gradient_estimators import full_gradient
from src.gradient_estimators import truncated_bptt
from src.gradient_estimators import full_es
from src.gradient_estimators import full_pesk

FLAGS = flags.FLAGS

def use_random_initial_iteration_for_truncated_step_train():
  # this is for creating the truncated step for training
  # these methods below don't need unlocked workers
  START_FROM_BEGINNING_METHODS = [
    "FullGradient",
    "FullDirectionalGradient",
    "ShortHorizonGradient",
    "FullES",
    "ShortHorizonES",
    "FullPESK",
    "FullUORO",]
  return \
    (FLAGS.gradient_estimator.upper() not in [x.upper() for x in START_FROM_BEGINNING_METHODS])

def smoothing_in_objective():
  # this is for running evaluation
  # these methods below don't need smoothing evaluations
  NO_SMOOTHING = [
    "FullGradient",
    "TruncatedForwardMode",
    "TruncatedBPTT",
    "ShortHorizonGradient",
    "FullDirectionalGradient",
    "TruncatedForwardModeK",
    "TruncatedForwardModeSharedNoise",
    "FullUORO",
    "UORO",]
  return \
    (FLAGS.gradient_estimator.upper() not in [x.upper() for x in NO_SMOOTHING])
    
def use_jitting_in_truncated_step():
  NO_JITTING = ["rl"]
  return (FLAGS.application.upper() not in [x.upper() for x in NO_JITTING])

def use_burn_in():
  # whether to use burn-in when running init_work_state for gradient estimator
  # NO_NEED_FOR_BURN_IN = ["rl"]
  # return (FLAGS.application.upper() not in [x.upper() for x in NO_NEED_FOR_BURN_IN])
  return True

def create_gradient_estimator(truncated_step,):
  # import ipdb; ipdb.set_trace()
  if use_burn_in():
    burn_in_length = FLAGS.horizon_length
  else:
    burn_in_length = 0

  use_jitting = use_jitting_in_truncated_step()
  # return a gradient estimator object
  if FLAGS.gradient_estimator.upper() == "TruncatedNRES".upper():
    grad_est = truncated_nres.TruncatedNRES(
        truncated_step=truncated_step,
        unroll_length=FLAGS.trunc_length,
        std=FLAGS.sigma,
        burn_in_length=burn_in_length)

  elif FLAGS.gradient_estimator.upper() == "TruncatedESBiased".upper():
    grad_est = truncated_es_biased.TruncatedESBiased(
      truncated_step=truncated_step,
      unroll_length=FLAGS.trunc_length,
      std=FLAGS.sigma,
      burn_in_length=burn_in_length,
      jitted=use_jitting)

  elif FLAGS.gradient_estimator.upper() == "TruncatedPESK".upper():
    grad_est = truncated_pesk.TruncatedPESK(
      truncated_step=truncated_step,
      unroll_length=FLAGS.trunc_length,
      K=FLAGS.K,
      std=FLAGS.sigma,
      burn_in_length=burn_in_length,
      loss_normalize=FLAGS.loss_normalize,
      jitted=use_jitting)

  elif FLAGS.gradient_estimator.upper() == "TruncatedForwardModeSharedNoise".upper():
    grad_est = truncated_forwardmode.TruncatedForwardModeSharedNoise(
      truncated_step=truncated_step,
      unroll_length=FLAGS.trunc_length,
      burn_in_length=burn_in_length)

  elif FLAGS.gradient_estimator.upper() == "TruncatedForwardModeK".upper():
    grad_est = truncated_forwardmodek.TruncatedForwardModeK(
      truncated_step=truncated_step,
      unroll_length=FLAGS.trunc_length,
      K=FLAGS.K,
      burn_in_length=burn_in_length)

  elif FLAGS.gradient_estimator.upper() == "UORO".upper():
    grad_est = uoro.UORO(
      truncated_step=truncated_step,
      unroll_length=FLAGS.trunc_length,
      burn_in_length=burn_in_length,
    )

  elif FLAGS.gradient_estimator.upper() == "FullUORO".upper():
    grad_est = uoro.FullUORO(
      truncated_step=truncated_step,
      T=FLAGS.horizon_length,
    )

  elif FLAGS.gradient_estimator.upper() == "FullDirectionalGradient".upper():
    # this version doesn't use K
    grad_est = full_directional_gradientk.FullDirectionalGradient(
      truncated_step=truncated_step,
      T=FLAGS.horizon_length,
    )

  elif FLAGS.gradient_estimator.upper() == "FullGradient".upper():
    grad_est = full_gradient.FullGradient(
      truncated_step=truncated_step,
      T=FLAGS.horizon_length,
    )
  
  elif FLAGS.gradient_estimator.upper() == "ShortHorizonGradient".upper():
    # only using the first FLAGS.trunc_length steps
    # this is in contrast to the TruncatedBPTT which still start from multiples of trunc_length
    grad_est = full_gradient.FullGradient(
      truncated_step=truncated_step,
      T=FLAGS.trunc_length,)
    # we overwrite the name property of the gradient estimator
    def name(self):
      return \
      ("ShortHorizonGradient"
      f"_N={self.truncated_step.num_tasks},W={self.T}")

    grad_est.grad_est_name = types.MethodType(name, grad_est)

  elif FLAGS.gradient_estimator.upper() == "TruncatedBPTT".upper():
    grad_est = truncated_bptt.TruncatedBPTT(
      truncated_step=truncated_step,
      unroll_length=FLAGS.trunc_length,
      burn_in_length=burn_in_length,
    )

  elif FLAGS.gradient_estimator.upper() == "FullES".upper():
    # change this for RL training
    # grad_est = full_es.FullES(
    if use_jitting:
      construct_fn = full_es.FullES
    else:
      construct_fn = full_es.FullES_unjitted
    grad_est = construct_fn(truncated_step=truncated_step,
        std=FLAGS.sigma,
        T=FLAGS.horizon_length,
        loss_normalize=FLAGS.loss_normalize,)

    # grad_est = full_pesk.FullPESK(
    #   truncated_step=truncated_step,
    #   K=horizon_length,
    #   std=FLAGS.sigma,
    #   T=horizon_length,
    #   loss_normalize=False,
    # )

  elif FLAGS.gradient_estimator.upper() == "ShortHorizonES".upper():
    # only using the first FLAGS.trunc_length steps
    if use_jitting:
      construct_fn = full_es.FullES
    else:
      construct_fn = full_es.FullES_unjitted
    grad_est = construct_fn(truncated_step=truncated_step,
                            std=FLAGS.sigma,
                            T=FLAGS.trunc_length,
                            loss_normalize=FLAGS.loss_normalize,)
    # we overwrite the grad_est_name function of the gradient estimator
    def name(self):
      # self.T will now be FLAGS.trunc_length
      return \
      ("ShortHorizonES"
      f"_N={self.truncated_step.num_tasks},W={self.T}")

    grad_est.grad_est_name = types.MethodType(name, grad_est)

  elif FLAGS.gradient_estimator.upper() == "FullPESK".upper():
    grad_est = full_pesk.FullPESK(
      truncated_step=truncated_step,
      K=FLAGS.K,
      std=FLAGS.sigma,
      T=FLAGS.horizon_length,
      loss_normalize=FLAGS.loss_normalize,
    )

  else:
    raise ValueError(f"gradient_estimator {FLAGS.gradient_estimator} not supported")
  
  return grad_est