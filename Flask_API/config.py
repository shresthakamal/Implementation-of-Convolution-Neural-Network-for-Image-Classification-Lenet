import os
import datetime
import numpy as np
import pathlib
import tensorflow as tf

AUTOTUNE = tf.data.experimental.AUTOTUNE

CURRENT_DIR = pathlib.Path(__file__)
BASE_DIR = CURRENT_DIR.parent.parent
DATASET_DIR = BASE_DIR/"datasets"

TRAIN_DIR = DATASET_DIR/"seg_train"
VALIDATION_DIR = DATASET_DIR/"seg_test"
TEST_DIR = DATASET_DIR/"seg_pred"


IMG_HEIGHT = 150
IMG_WIDTH = 150

LEARNING_RATE = 0.001
EPOCHS = 10
BATCH_SIZE = 32

TRAIN_LOG_DIR = f'./temp/train/{datetime.datetime.now()}/logs'
CHECKPOINT_DIR = f'./temp/train/checkpoints/'
pathlib.Path(CURRENT_DIR.parent/"temp/models").mkdir(parents=True, exist_ok=True)
MODEL_DIR = CURRENT_DIR.parent/"temp/models"
WEIGHTS_PATH = os.path.join(str(MODEL_DIR), 'weights.h5')

OPTIMIZER = tf.optimizers.Adam(learning_rate=LEARNING_RATE)
CATEGORICAL_LOSS = tf.losses.CategoricalCrossentropy(from_logits=True)