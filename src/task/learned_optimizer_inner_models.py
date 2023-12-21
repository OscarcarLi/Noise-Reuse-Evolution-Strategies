from typing import Sequence, List, Callable
import jax
import jax.numpy as jnp
import haiku


class MLP(haiku.Module):
  def __init__(
    self,
    num_classes: int,
    activation: Callable[[jax.Array], jax.Array], # activation function for the MLP e.g. jax.nn.relu
    hidden_dims: List[int], # the hidden dimensions of the MLP which doesn't count the class dimension
    dropout_rate: float = 0.0,
  ):
    super().__init__(name="MLP")
    self.num_classes = num_classes
    self.activation = activation
    self.hidden_dims = hidden_dims
    self.dropout_rate = dropout_rate
    self.reshape = haiku.Reshape((-1,), preserve_dims=1)
    self.mlp = haiku.nets.MLP(
        self.hidden_dims + [self.num_classes],
        activation=self.activation)

  def name(self) -> str:
    return f"MLP{'-'.join([str(x) for x in self.hidden_dims])}_{self.activation.__name__}_dropout{self.dropout_rate}"

  def __call__(self, X) -> jax.Array:
    # X: [batch_size, ...]
    # return logits of shape [batch_size, num_classes]
    value = self.reshape(X)
    logits = self.mlp(value,
                      dropout_rate=self.dropout_rate, rng=haiku.next_rng_key())
    return logits


class FullConvNet(haiku.Module):
  def __init__(
    self,
    num_classes: int,
    hidden_channels: List[int],
    activation : Callable[[jax.Array], jax.Array]=jax.nn.relu,
  ):

    super().__init__(name="FullConvNet")
    self.layers = []
    self.num_classes = num_classes
    self.hidden_channels = hidden_channels
    self.activation = activation
    for n_channels in hidden_channels:
      # the initialization isn't exactly the same as that in pytorch
      conv = haiku.Conv2D(
                output_channels=n_channels,
                kernel_shape=(3, 3),
                # stride=(2, 2),
                padding="SAME")
      self.layers.extend([conv, self.activation,])
      # instance_norm = haiku.InstanceNorm(create_scale=True, create_offset=True)
      avg_pool = haiku.AvgPool(window_shape=2, strides=2, padding="VALID",)
      self.layers.append(avg_pool)

      # self.layers.extend([conv, instance_norm, self.activation, avg_pool])

    self.layers.append(haiku.Reshape((-1,), preserve_dims=1))
    self.layers.append(haiku.Linear(self.num_classes))

  def name(self) -> str:
    return (
      f"FullConvNet{'-'.join([str(x) for x in self.hidden_channels])}_"
      f"{self.activation.__name__}"
    )
    
  def __call__(self, X: jax.Array) -> jax.Array:
    # X: [batch_size, ...]
    # return logits of shape [batch_size, num_classes]
    value = X
    for layer in self.layers:
      value = layer(value)
      # TESTED  the dimension seems to be correct
      # print(x.shape)
    return value
