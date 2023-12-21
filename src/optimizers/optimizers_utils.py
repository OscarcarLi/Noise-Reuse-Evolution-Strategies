import optax
from src.optimizers import base as opt_base

def create_optimizer(specs: str):
  """given an optimizer specification string
      return an optax optimizer object

  Args:
      specs (str):
        for example,
          * SGD||CONSTANT1e-5
          * currently for SGDM, we only support moment of 0.9
          * ADAM||PIECEWISE_CONSTANT1e-5(150,0.3)(200,0.333333333333333)
  
  Returns: 
      optax.GradientTransformation: the optax optimizer specified by specs
  """
  SGD = "SGD"
  SGDM = "SGDM"
  ADAM = "ADAM"
  SEPARATOR = "||"
  
  optimizer_type_spec = specs[:specs.find(SEPARATOR)]
  learning_rate_schedule_fn = create_learning_rate_schedule_fn(
    lr_specs=specs[specs.find(SEPARATOR) + len(SEPARATOR):]
  )
  
  if optimizer_type_spec.upper() == SGD:
    return opt_base.SGD(learning_rate=learning_rate_schedule_fn)
  elif optimizer_type_spec.upper() == SGDM:
    return opt_base.SGDM(learning_rate=learning_rate_schedule_fn)
  elif optimizer_type_spec.upper() == ADAM:
    return opt_base.Adam(learning_rate=learning_rate_schedule_fn)
  else:
    raise ValueError("optimizer type not supported")


def create_learning_rate_schedule_fn(lr_specs: str):
  """given a learning rate specification string
    return a learning rate mapping function

  Args:
      lr_specs (str): 
      for example,
        * CONSTANT1e-5
        * PIECEWISE_CONSTANT1e-5(150,0.3)(200,0.333333333333333)

  Returns:
      Callable[int: float]: a function that maps integer iteration number
        to the corresponding learning rate value
  """
  CONSTANT = "CONSTANT"
  PIECEWISE_CONSTANT = "PIECEWISE_CONSTANT"

  if lr_specs.upper().startswith(CONSTANT):
    lr_specs = lr_specs[len(CONSTANT):]
    return optax.constant_schedule(value=float(lr_specs))

  elif lr_specs.upper().startswith(PIECEWISE_CONSTANT):
    lr_specs = lr_specs[len(PIECEWISE_CONSTANT):]
    init_value = float(lr_specs[:lr_specs.find("(")])
    lr_specs = lr_specs[lr_specs.find("("):]
    boundaries_and_scales = {}
    lr_piece_list = lr_specs.split(")(")
    for lr_piece in lr_piece_list:
      # strip of "(" and ")"
      lr_piece = lr_piece.strip("()")
      assert len(lr_piece.split(",")) == 2
      boundary, ratio = lr_piece.split(",")
      boundary = int(boundary)
      ratio = float(ratio)
      boundaries_and_scales[boundary] = ratio

    return optax.piecewise_constant_schedule(
        init_value=init_value,
        boundaries_and_scales=boundaries_and_scales,
      )
  else:
    raise ValueError(f"Not supported learning rate schedule {lr_specs}")


if __name__ == "__main__":
  lr_tests = [
    "constant1e-5",
    "piecewise_constant1e-5(150,0.3)(200,0.333333333333333)"
  ]

  for t in lr_tests:
    print(t)
    fn = create_learning_rate_schedule_fn(t)
    for i in range(2000):
      print(i, fn(i))
    print("\n"*100)
  

  opt_tests = [
    "SGD||CONSTANT1e-5",
    "ADAM||PIECEWISE_CONSTANT1e-5(150,0.3)(200,0.333333333333333)",
  ]
  for t in opt_tests:
    print(t)
    opt = create_optimizer(t)
    print(opt)