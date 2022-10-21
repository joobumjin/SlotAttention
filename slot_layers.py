from absl import logging
import tensorflow as tf
import matplotlib.pyplot as plt
import collections
import numpy as np
from tqdm import tqdm

"""Slot Attention model for object discovery and set prediction."""

class SlotAttention(tf.keras.layers.Layer):
  """Slot Attention module."""

  def __init__(self, num_iterations, num_slots, slot_size, mlp_hidden_size,
               epsilon=1e-8):
    """Builds the Slot Attention module.
    Args:
      num_iterations: Number of iterations.
      num_slots: Number of slots.
      slot_size: Dimensionality of slot feature vectors.
      mlp_hidden_size: Hidden layer size of MLP.
      epsilon: Offset for attention coefficients before normalization.
    """
    super().__init__()
    self.num_iterations = num_iterations
    self.num_slots = num_slots
    self.slot_size = slot_size
    self.mlp_hidden_size = mlp_hidden_size
    self.epsilon = epsilon

    self.norm_inputs = tf.keras.layers.LayerNormalization()
    self.norm_slots = tf.keras.layers.LayerNormalization()
    self.norm_mlp = tf.keras.layers.LayerNormalization()

    # Parameters for Gaussian init (shared by all slots).   # Intialize slots randomly at first 
    self.slots_mu = self.add_weight(
        initializer="glorot_uniform",
        shape=[1, 1, self.slot_size],   # slot_size: Dimensionality of slot feature vectors.
        dtype=tf.float32,
        name="slots_mu")
    self.slots_log_sigma = self.add_weight(
        initializer="glorot_uniform",
        shape=[1, 1, self.slot_size],
        dtype=tf.float32,
        name="slots_log_sigma")

    # Linear maps for the attention module.
    self.project_q = tf.keras.layers.Dense(self.slot_size, use_bias=False, name="q")
    self.project_k = tf.keras.layers.Dense(self.slot_size, use_bias=False, name="k")
    self.project_v = tf.keras.layers.Dense(self.slot_size, use_bias=False, name="v")

    # Slot update functions.
    self.gru = tf.keras.layers.GRUCell(self.slot_size)
    self.mlp = tf.keras.Sequential([
        tf.keras.layers.Dense(self.mlp_hidden_size, activation="relu"),
        tf.keras.layers.Dense(self.slot_size)
    ], name="mlp")

  def call(self, inputs):
    # `inputs` has shape [batch_size, num_inputs, inputs_size].
    inputs = self.norm_inputs(inputs)  # Apply layer norm to the input.
    k = self.project_k(inputs)  # Shape: [batch_size, num_inputs, slot_size].  # create key vectors (based on inputs)
    v = self.project_v(inputs)  # Shape: [batch_size, num_inputs, slot_size].  # create value vectors (based on inputs)

    # Initialize the slots. Shape: [batch_size, num_slots, slot_size].
    slots = self.slots_mu + tf.exp(self.slots_log_sigma) * tf.random.normal(
        [tf.shape(inputs)[0], self.num_slots, self.slot_size])  # size: [batch_size, num_slots, slot_size]

    # Multiple rounds of attention.
    for _ in range(self.num_iterations):
      slots_prev = slots
      slots = self.norm_slots(slots)

      # Attention.
      q = self.project_q(slots)  # Shape: [batch_size, num_slots, slot_size].  # create query vectors (based on slots)
      q *= self.slot_size ** -0.5  # Normalization.
      attn_logits = tf.keras.backend.batch_dot(k, q, axes=-1) # Batchwise dot product.
      attn = tf.nn.softmax(attn_logits, axis=-1)
      # `attn` has shape: [batch_size, num_inputs, num_slots]. 
      # attn represents how much attention each slot should pay to the features 

      # Weigted mean.
      attn += self.epsilon
      attn /= tf.reduce_sum(attn, axis=-2, keepdims=True) # summation; sum across the batch_size 
      updates = tf.keras.backend.batch_dot(attn, v, axes=-2)
      # `updates` has shape: [batch_size, num_slots, slot_size].

      # Slot update.
      slots, _ = self.gru(updates, [slots_prev])   # output after gru has shape: [batch_size, num_slots, slot_size]
      slots += self.mlp(self.norm_mlp(slots))      # # output after mlp has shape: [batch_size, num_slots, slot_size]

    return slots


def spatial_broadcast(slots, resolution):
  """Broadcast slot features to a 2D grid and collapse slot dimension."""
  # `slots` has shape: [batch_size, num_slots, slot_size].
  slots = tf.reshape(slots, [-1, slots.shape[-1]])[:, None, None, :]
  grid = tf.tile(slots, [1, resolution[0], resolution[1], 1])   # this operation creates a new tensor by replicating input multiples times
  # `grid` has shape: [batch_size*num_slots, width, height, slot_size].
  return grid


def spatial_flatten(x):
  return tf.reshape(x, [-1, x.shape[1] * x.shape[2], x.shape[-1]])


def unstack_and_split(x, batch_size, num_channels=3):
  """Unstack batch dimension and split into channels and alpha mask."""
  unstacked = tf.reshape(x, [batch_size, -1] + x.shape.as_list()[1:])
  channels, masks = tf.split(unstacked, [num_channels, 1], axis=-1)
  return channels, masks
    

def build_grid(resolution):
  ranges = [np.linspace(0., 1., num=res) for res in resolution]
  grid = np.meshgrid(*ranges, sparse=False, indexing="ij")
  grid = np.stack(grid, axis=-1)
  grid = np.reshape(grid, [resolution[0], resolution[1], -1])
  grid = np.expand_dims(grid, axis=0)
  grid = grid.astype(np.float32)
  return np.concatenate([grid, 1.0 - grid], axis=-1)


class SoftPositionEmbed(tf.keras.layers.Layer):
  """Adds soft positional embedding with learnable projection."""

  def __init__(self, hidden_size, resolution):
    """Builds the soft position embedding layer.
    Args:
      hidden_size: Size of input feature dimension.
      resolution: Tuple of integers specifying width and height of grid.
    """
    super().__init__()
    self.dense = tf.keras.layers.Dense(hidden_size, use_bias=True)
    self.grid = build_grid(resolution)

  def call(self, inputs):
    return inputs + self.dense(self.grid)