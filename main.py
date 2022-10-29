from absl import logging
import tensorflow as tf
import matplotlib.pyplot as plt
import collections
import numpy as np
import tensorflow.keras.layers as layers
from tqdm import tqdm
from data_utils import *
from model import *

"""Training loop for object discovery with Slot Attention."""

# We use `tf.function` compilation to speed up execution. For debugging,
# consider commenting out the `@tf.function` decorator.


def l2_loss(prediction, target):
  return tf.reduce_mean(tf.math.squared_difference(prediction, target))


@tf.function
def train_step(batch, model, optimizer):
  """Perform a single training step."""

  # Get the prediction of the models and compute the loss.
  with tf.GradientTape() as tape:
    preds = model(batch["image"], training=True)
    recon_combined, recons, masks, slots = preds
    loss_value = l2_loss(recon_combined, batch["image"])
    del recons, masks, slots  # Unused.

  # Get and apply gradients.
  gradients = tape.gradient(loss_value, model.trainable_weights)
  optimizer.apply_gradients(zip(gradients, model.trainable_weights))   

  return loss_value


def visualize_loss(losses): 
  """
  Uses Matplotlib to visualize the losses of our model.
  :param losses: list of loss data stored from train. Can use the model's loss_list 
  field 

  NOTE: DO NOT EDIT

  :return: doesn't return anything, a plot should pop-up 
  """
  x = [i for i in range(len(losses))]
  plt.plot(x, losses)
  plt.title('Loss per epoch')
  plt.xlabel('Training Epoch')
  plt.ylabel('Loss')
  plt.show()

def renormalize(x):
  """Renormalize from [-1, 1] to [0, 1]."""
  return x / 2. + 0.5

def get_prediction(model, batch, idx=0):
  recon_combined, recons, masks, slots = model(batch["image"])
  image = renormalize(batch["image"])[idx]
  recon_combined = renormalize(recon_combined)[idx]
  recons = renormalize(recons)[idx]
  masks = masks[idx]
  return image, recon_combined, recons, masks, slots

def main():
  # Hyperparameters of the model.
  batch_size = 64
  num_slots = 7
  num_iterations = 3
  base_learning_rate = 0.0004
  num_train_steps = 100
  warmup_steps = 5
  decay_rate = 0.5
  decay_steps = 100000
  tf.random.set_seed(0)
  resolution = (256, 256)

  # Build dataset iterators, optimizers and model.
  data_iterator = allen_cell_dataset(False, batch_size)

  optimizer = tf.keras.optimizers.Adam(base_learning_rate, epsilon=1e-08)

  model = build_model(resolution, batch_size, num_slots,
                      num_iterations, model_type="object_discovery")
    
  # Prepare checkpoint manager.
  global_step = tf.Variable(
      0, trainable=False, name="global_step", dtype=tf.int64)

  losses = []

  for _ in tqdm(range(num_train_steps), desc='Training Epochs'):
      batch = next(data_iterator)

      # Learning rate warm-up.
      if global_step < warmup_steps:
        learning_rate = base_learning_rate * tf.cast(
            global_step, tf.float32) / tf.cast(warmup_steps, tf.float32)
      else:
        learning_rate = base_learning_rate
      
      learning_rate = learning_rate * (decay_rate ** (
          tf.cast(global_step, tf.float32) / tf.cast(decay_steps, tf.float32)))
      optimizer.lr = learning_rate.numpy()

      loss_value = train_step(batch, model, optimizer)
      losses.append(loss_value)

      # Update the global step. We update it before logging the loss and saving
      # the model so that the last checkpoint is saved at the last iteration.
      global_step.assign_add(1)

  visualize_loss(losses)

  data_iterator = allen_cell_dataset(False, batch_size)

  batch = next(data_iterator)

  image, recon_combined, recons, masks, slots = get_prediction(model, batch)

  # Visualize.
  num_slots = len(masks)
  fig, ax = plt.subplots(1, num_slots + 2, figsize=(15, 2))
  ax[0].imshow(image)
  ax[0].set_title('Image')
  ax[1].imshow(recon_combined)
  ax[1].set_title('Recon.')
  for i in range(num_slots):
    ax[i + 2].imshow(recons[i] * masks[i] + (1 - masks[i]))
    ax[i + 2].set_title('Slot %s' % str(i + 1))
  for i in range(len(ax)):
    ax[i].grid(False)
    ax[i].axis('off')

if __name__ == main():
  main()