import io
import tifffile
import quilt3 as q3
import matplotlib.pyplot as plt
import tensorflow as tf
import collections
import numpy as np
from tqdm import tqdm
from aicsimageio import AICSImage #=> this package was really difficult to install, maybe using an automated yaml would be good
from PIL import Image
import os
from urllib.parse import urlparse, unquote
from os import listdir
from os.path import join

def fetch_data():
  package = q3.Package.browse(
      "aics/pipeline_integrated_single_cell",
      registry="s3://allencell"
  )

  package["cell_images_2d"].fetch("./AllenCell/cell_images_2d/")

  return None


def allen_cell_dataset(download_data = False, batch_size = 64):
  # 70 train: 35k 
  # 20 validation: 10k
  # 10 test: 5k
  if download_data:
    fetch_data()

  def convert_to_padded_tensor(img):
      image_tensor = tf.convert_to_tensor(img.data[0][0])
      padded_tensor = tf.image.resize_with_crop_or_pad(image_tensor, 256, 256)
      return padded_tensor


  imgs = []
  file_names = [join("./AllenCell/cell_images_2d/", f) for f in listdir("./AllenCell/cell_images_2d/") if join("./AllenCell/cell_images_2d/", f).endswith(".png")]

  if len(file_names) == 0:
    raise Exception("No .png Files in the AllenCell directory.")

  for file_name in file_names:
    img = AICSImage(file_name)

    tensor = convert_to_padded_tensor(img)
    imgs.append(tensor[0])

  ## split into train validation test

  dataset = tf.data.Dataset.from_tensor_slices(imgs)

  dataset = dataset.shuffle()

  test_val = dataset.take(15000)
  train = dataset.skip(150000)
  train = train.batch(batch_size)

  test = test_val.take(5000)
  test = test.batch(batch_size)
  val = test_val.take(10000)
  val = val.batch(batch_size)

  return train, test, val