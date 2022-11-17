from absl import logging
import tensorflow as tf
import matplotlib.pyplot as plt
import collections
import numpy as np
from tqdm import tqdm
from data_utils import *
from model import *
from itertools import cycle

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
  x = [i for i in range(len(losses))]
  plt.plot(x, losses)
  plt.title('Loss per epoch')
  plt.xlabel('Training Epoch')
  plt.ylabel('Loss')
  plt.show()

def save_loss(losses, file_name):
  x = [i for i in range(len(losses))]
  plt.plot(x, losses)
  plt.title('Loss per epoch')
  plt.xlabel('Training Epoch')
  plt.ylabel('Loss')
  plt.savefig(file_name)

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
  batch_size = 8
  num_slots = 7
  num_iterations = 3
  base_learning_rate = 0.0004
  num_train_steps = 2000
  warmup_steps = 40
  decay_rate = 0.5
  decay_steps = 100000
  #tf.random.set_seed(0)
  resolution = (256, 256)

  checkpoint_path = "./training/cp-{epoch}.ckpt"

  # Build dataset iterators, optimizers and model.
  train_iterator, test_iterator, val_iterator = allen_cell_dataset(False, batch_size)
  train_iterator = cycle(list(train_iterator))
  test_iterator = cycle(list(test_iterator))
  val_iterator = cycle(list(val_iterator))

  optimizer = tf.keras.optimizers.Adam(base_learning_rate, epsilon=1e-08)

  model = build_model(resolution, batch_size, num_slots, num_iterations)
    
  # Prepare checkpoint manager.
  global_step = tf.Variable(
      0, trainable=False, name="global_step", dtype=tf.int64)

  losses = []

  for _ in tqdm(range(num_train_steps), desc='Training Epochs'):
      batch = next(train_iterator)
      val_batch = next(val_iterator)

      # Learning rate warm-up.
      if global_step < warmup_steps:
          learning_rate = base_learning_rate * tf.cast(global_step, tf.float32) / tf.cast(warmup_steps, tf.float32)
      else:
          learning_rate = base_learning_rate

      learning_rate = learning_rate * (decay_rate ** (tf.cast(global_step, tf.float32) / tf.cast(decay_steps, tf.float32)))
      optimizer.lr = learning_rate.numpy()

      loss_value = train_step(batch, model, optimizer)
      losses.append(loss_value)
      
      val_recon_combined, _, _, _ = model(val_batch, training=False)
      val_losses.append(l2_loss(val_recon_combined, val_batch))

      # Update the global step. We update it before logging the loss and saving
      # the model so that the last checkpoint is saved at the last iteration.
      global_step.assign_add(1)

  model.save_weights(checkpoint_path.format(epoch=num_train_steps))
  model.save_loss(losses, f"training_loss_{num_train_steps}.png")
  model.save_loss(val_losses, f"validation_loss_{num_train_steps}.png")

  #visualize_loss(losses)

  """
  batch = next(test_iterator)

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

  """

if __name__ == main():
  main()