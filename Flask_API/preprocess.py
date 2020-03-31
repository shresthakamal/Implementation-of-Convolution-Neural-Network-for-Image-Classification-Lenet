import tensorflow as tf
import numpy as np
import sys
import os
from PIL import Image
sys.path.append(os.pardir)
sys.path.append(os.path.join(os.pardir, os.pardir))

from src.config import (
                        TRAIN_DIR,
                        VALIDATION_DIR,
                        TEST_DIR,
                        BATCH_SIZE,
                        IMG_HEIGHT,
                        IMG_WIDTH,
                        AUTOTUNE
)

CLASS_NAMES = np.array([item.name for item in TRAIN_DIR.glob('*')])
print(CLASS_NAMES)

def get_label(file_path):
  # convert the path to a list of path components
  parts = tf.strings.split(file_path, os.path.sep)
  # The second to last is the class-directory
  class_labels = parts[-2] == CLASS_NAMES
  class_labels = tf.dtypes.cast(class_labels, tf.float32)
  return class_labels


def decode_img(img):
  # convert the compressed string to a 3D uint8 tensor
  img = tf.image.decode_jpeg(img, channels=3)
  # Use `convert_image_dtype` to convert to floats in the [0,1] range.
  img = tf.image.convert_image_dtype(img, tf.float32)
  # resize the image to the desired size.
  return tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])


def process_path(file_path):
  label = get_label(file_path)
  # load the raw data from the file as a string
  img = tf.io.read_file(file_path)
  img = decode_img(img)
  return img, label


def prepare_for_training(ds, cache=True, shuffle_buffer_size=1000):
  # This is a small dataset, only load it once, and keep it in memory.
  # use `.cache(filename)` to cache preprocessing work for datasets that don't
  # fit in memory.
  if cache:
    if isinstance(cache, str):
      ds = ds.cache(cache)
    else:
      ds = ds.cache()

  ds = ds.shuffle(buffer_size=shuffle_buffer_size)

  # Repeat forever
  # ds = ds.repeat()

  ds = ds.batch(BATCH_SIZE)

  # `prefetch` lets the dataset fetch batches in the background while the model
  # is training.
  ds = ds.prefetch(buffer_size=AUTOTUNE)

  return ds

def process_single_image(TEST_IMG_PATH):
    img = tf.io.read_file(TEST_IMG_PATH)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])
    img = tf.expand_dims(img, axis=0)
    return img

def convert_image(file):
    img = Image.open(file)
    img = np.array(img)
    img = tf.convert_to_tensor(
        img, dtype=None, dtype_hint=None, name=None
    )
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])
    img = tf.expand_dims(img, axis=0)
    return img

