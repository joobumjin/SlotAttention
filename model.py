from absl import logging
import tensorflow as tf
import matplotlib.pyplot as plt
import collections
import numpy as np
from tqdm import tqdm

import slot_layers

def build_model(resolution, batch_size, num_slots, num_iterations,
                num_channels=3, model_type="object_discovery"):
  """Build keras model."""

  encoder_cnn = tf.keras.Sequential([
      tf.keras.layers.Conv2D(64, kernel_size=5, padding="SAME", activation="relu"),
      tf.keras.layers.Conv2D(64, kernel_size=5, padding="SAME", activation="relu"),
      tf.keras.layers.Conv2D(64, kernel_size=5, padding="SAME", activation="relu"),
      tf.keras.layers.Conv2D(64, kernel_size=5, padding="SAME", activation="relu")
  ], name="encoder_cnn")

  decoder_initial_size = (8, 8)
  decoder_cnn = tf.keras.Sequential([
      tf.keras.layers.Conv2DTranspose(64, 5, strides=(2, 2), padding="SAME", activation="relu"),  
      tf.keras.layers.Conv2DTranspose(64, 5, strides=(2, 2), padding="SAME", activation="relu"),
      tf.keras.layers.Conv2DTranspose(64, 5, strides=(2, 2), padding="SAME", activation="relu"),
      tf.keras.layers.Conv2DTranspose(64, 5, strides=(2, 2), padding="SAME", activation="relu"),
      tf.keras.layers.Conv2DTranspose(64, 5, strides=(2, 2), padding="SAME", activation="relu"),
      tf.keras.layers.Conv2DTranspose(4, 3, strides=(1, 1), padding="SAME", activation=None)
  ], name="decoder_cnn")

  encoder_pos = slot_layers.SoftPositionEmbed(64, resolution)
  decoder_pos = slot_layers.SoftPositionEmbed(64, decoder_initial_size)

  layer_norm = tf.keras.layers.LayerNormalization()
  mlp = tf.keras.Sequential([
      tf.keras.layers.Dense(64, activation="relu"),
      tf.keras.layers.Dense(64)
  ], name="encoded_feedforward")

  slot_attention = slot_layers.SlotAttention(num_iterations=num_iterations, num_slots=num_slots, slot_size=64, mlp_hidden_size=128)

  # Convolutional encoder with position embedding.
  inputs = tf.keras.Input(shape=(256,256,3,))
  x = encoder_cnn(inputs)  # CNN Backbone.
  x = encoder_pos(x)  # Add positional embeddings to x
  x = slot_layers.spatial_flatten(x)  # Flatten spatial dimensions (treat image as set).
  x = mlp(layer_norm(x))  # Feedforward network on set.
  # `x` has shape: [batch_size, width*height, input_size(64)].

  # Slot Attention module.
  slots = slot_attention(x)
  # `slots` has shape: [batch_size, num_slots, slot_size].

  # Spatial broadcast decoder.
  x = slot_layers.spatial_broadcast(slots, decoder_initial_size)
  # `x` has shape: [batch_size*num_slots, width_init, height_init, slot_size].
  x = decoder_pos(x)
  x = decoder_cnn(x)
  # `x` has shape: [batch_size*num_slots, width, height, num_channels+1].

  # Undo combination of slot and batch dimension; split alpha masks.
  recons, masks = slot_layers.unstack_and_split(x, batch_size=batch_size)
  # `recons` has shape: [batch_size, num_slots, width, height, num_channels].
  # `masks` has shape: [batch_size, num_slots, width, height, 1].

  # Normalize alpha masks over slots.
  masks = tf.nn.softmax(masks, axis=1)
  recon_combined = tf.reduce_sum(recons * masks, axis=1)  # Recombine image.
  # `recon_combined` has shape: [batch_size, width, height, num_channels].

  outputs = recon_combined, recons, masks, slots

  slot_attention_ae = tf.keras.Model(inputs = inputs, outputs = outputs, name="Slot_Attention_AutoEnconder")
  slot_attention_ae.summary()
  return slot_attention_ae  