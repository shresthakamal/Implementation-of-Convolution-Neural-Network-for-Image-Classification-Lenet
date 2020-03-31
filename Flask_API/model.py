import tensorflow as tf
import sys
import os
sys.path.append(os.pardir)
sys.path.append(os.path.join(os.pardir, os.pardir))
from src.preprocess import process_path, prepare_for_training

from src.config import (
    TRAIN_DIR,
    VALIDATION_DIR,
    EPOCHS,
    AUTOTUNE,
    TRAIN_LOG_DIR,
    CHECKPOINT_DIR,
    WEIGHTS_PATH,
    OPTIMIZER,
    CATEGORICAL_LOSS,
)

LOSSES = tf.keras.metrics.Mean(name='loss')
VALIDATION_LOSS = tf.keras.metrics.Mean(name='val_loss')

class AlexNet(tf.keras.Model):
    def __init__(self):
        super(AlexNet, self).__init__()
        initializer = tf.initializers.GlorotUniform(seed=123)

        # Conv1
        # filter size = [f_h, f_w, f_c, num_channel]
        self.wc1 = tf.Variable(initializer([9, 9, 3, 64]), trainable=True, name='conv1')

        # Conv2
        # filter size = [f_h, f_w, f_c, num_channel]
        self.wc2 = tf.Variable(initializer([5, 5, 64, 32]), trainable=True, name='conv2')

        # Conv3
        # filter size = [f_h, f_w, f_c, num_channel]
        self.wc3 = tf.Variable(initializer([3, 3, 32, 16]), trainable=True, name='conv3')

        # Flatten

        # Dense Layers
        self.wd1 = tf.Variable(initializer([3136, 512]), trainable=True, name='dense1')
        self.wd2 = tf.Variable(initializer([512, 256]), trainable=True, name='dense2')
        self.wd3 = tf.Variable(initializer([256, 64]), trainable=True, name='dense3')
        self.wd4 = tf.Variable(initializer([64, 6]), trainable=True, name='dense4')

        #Conv biases
        self.bc1 = tf.Variable(tf.zeros([64]), dtype=tf.float32, trainable=True, name='conv1_bias')
        self.bc2 = tf.Variable(tf.zeros([32]), dtype=tf.float32, trainable=True, name='conv2_bias')
        self.bc3 = tf.Variable(tf.zeros([16]), dtype=tf.float32, trainable=True, name='conv3_bias')

        #Dense biases
        self.bd1 = tf.Variable(tf.zeros([512]), dtype=tf.float32, trainable=True, name='dense1_bias')
        self.bd2 = tf.Variable(tf.zeros([256]), dtype=tf.float32, trainable=True, name='dense2_bias')
        self.bd3 = tf.Variable(tf.zeros([64]), dtype=tf.float32, trainable=True, name='dense3_bias')
        self.bd4 = tf.Variable(tf.zeros([6]), dtype=tf.float32, trainable=True, name='dense4_bias')

    def call(self, x):
        # X = NHWC
        # Conv1 + maxpool 1
        x = tf.nn.conv2d(x, self.wc1, strides=[1, 3, 3, 1], padding="VALID")
        x = tf.nn.bias_add(x, self.bc1)
        x = tf.nn.relu(x)
        # print(f"CONV 1:{x.shape}")
        x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")
        # print(f"MAX POOL 1:{x.shape}")

        # Conv2 + maxpool 2
        x = tf.nn.conv2d(x, self.wc2, strides=[1, 1, 1, 1], padding="VALID")
        x = tf.nn.bias_add(x, self.bc2)
        x = tf.nn.relu(x)
        # print(f"CONV 2:{x.shape}")
        x = tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding="VALID")
        # print(f"MAX POOL 2:{x.shape}")

        # Conv2 + maxpool 3
        x = tf.nn.conv2d(x, self.wc3, strides=[1, 1, 1, 1], padding="VALID")
        x = tf.nn.bias_add(x, self.bc3)
        x = tf.nn.relu(x)
        # print(f"CONV 3:{x.shape}")
        x = tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding="VALID")
        # print(f"MAX POOL 3:{x.shape}")

        # Flattten out
        # N X Number of Nodes
        # Flatten()
        # print(f"FLATTEN LAYER: {tf.shape(x)[0]}")
        x = tf.reshape(x, (tf.shape(x)[0], -1))
        # print(x.shape)

        # Dense1
        x = tf.matmul(x, self.wd1)
        x = tf.nn.bias_add(x, self.bd1)
        x = tf.nn.relu(x)

        # Dense2
        x = tf.matmul(x, self.wd2)
        x = tf.nn.bias_add(x, self.bd2)
        x = tf.nn.relu(x)

        # Dense3
        x = tf.matmul(x, self.wd3)
        x = tf.nn.bias_add(x, self.bd3)

        # Dense4
        x = tf.matmul(x, self.wd4)
        x = tf.nn.bias_add(x, self.bd4)
        # x = tf.nn.sigmoid(x)
        # print(f"FINAL: {x.shape}")

        return x


def train_step(model, inputs, labels, loss_fn, optimzer):
    with tf.GradientTape() as t:
        y_predicted = model(inputs, training=True)
        current_loss = loss_fn(labels, y_predicted)

    gradients = t.gradient(current_loss, model.trainable_variables)
    OPTIMIZER.apply_gradients(zip(gradients, model.trainable_variables))
    return current_loss


def validation_step(model, inputs, labels, loss_fn):
    y_predicted = model(inputs, training=False)
    current_loss = loss_fn(labels, y_predicted)
    return current_loss


def check_for_checkpoint(manager):
    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print(f"restored from {manager.latest_checkpoint}")
    else:
        print("Initializing from scratch")


if __name__ == "__main__":
    train_ds = tf.data.Dataset.list_files(str(TRAIN_DIR / '*/*'))
    validation_ds = tf.data.Dataset.list_files(str(VALIDATION_DIR / '*/*'))

    labeled_train_ds = train_ds.map(process_path, num_parallel_calls=AUTOTUNE)
    labeled_validation_ds = validation_ds.map(process_path, num_parallel_calls=AUTOTUNE)

    train_dataset = prepare_for_training(labeled_train_ds)
    validation_dataset = prepare_for_training(labeled_validation_ds)

    model = AlexNet()
    file_writer = tf.summary.create_file_writer(TRAIN_LOG_DIR)
    ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=OPTIMIZER, net=model)
    manager = tf.train.CheckpointManager(ckpt, CHECKPOINT_DIR, max_to_keep=3)
    check_for_checkpoint(manager)
    for epoch in range(EPOCHS):
        ckpt.step.assign_add(1)

        print(f'epoch: {epoch}')
        LOSSES.reset_states()
        VALIDATION_LOSS.reset_states()
        for x_batch, y_batch in train_dataset:
            loss = train_step(model, x_batch, y_batch, CATEGORICAL_LOSS, OPTIMIZER)
            LOSSES(loss)

        save_path = manager.save()
        print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))

        with file_writer.as_default():
            tf.summary.scalar('loss', LOSSES.result(), step=epoch)

        print(LOSSES.result())

        for x_batch, y_batch in validation_dataset:
            val_loss = validation_step(model, x_batch, y_batch, CATEGORICAL_LOSS)
            VALIDATION_LOSS(val_loss)

        with file_writer.as_default():
            tf.summary.scalar('val_loss', VALIDATION_LOSS.result(), step=epoch)

        model.save_weights(WEIGHTS_PATH)
