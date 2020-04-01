import tensorflow as tf
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class LeNet(tf.keras.Model):
    def __init__(self):
        super (LeNet, self).__init__()
        initializer = tf.initializers.GlorotUniform(seed=123)
        #Conv1
        self.wc1 = tf.Variable(initializer([3,3,1,10]), trainable=True, name='wc1')
        #Conv2
        self.wc2 = tf.Variable(initializer([3,3,10,20]), trainable=True, name='wc2')
        
        #Flattening
        
        #Dense
        #Adjusting the weights for the Dense Layer (ANN)
        self.wd3 = tf.Variable(initializer([500,128]), trainable=True)
        self.wd4 = tf.Variable(initializer([128,64]), trainable=True)
        self.wd5 = tf.Variable(initializer([64,10]), trainable=True)
        
        #Adjusting the Biases
        #Biases for the Covolution Layers
        self.bc1 = tf.Variable(tf.zeros([10]), dtype=tf.float32, trainable=True)
        self.bc2 = tf.Variable(tf.zeros([20]), dtype=tf.float32, trainable=True)
        
        #Biases for the Dense Layers
        self.bd3 = tf.Variable(tf.zeros([128]), dtype=tf.float32, trainable=True)
        self.bd4 = tf.Variable(tf.zeros([64]), dtype=tf.float32, trainable=True)
        self.bd5 = tf.Variable(tf.zeros([10]), dtype=tf.float32, trainable=True)
        
    def call(self,x):
        
        # 1st Convolutional Layer
        #layer equals NHWC
        #First the Convolutional layer and the MaxPool
        x = tf.nn.conv2d(x, self.wc1, strides=[1,1,1,1], padding="VALID")
        x = tf.nn.bias_add(x,self.bc1)
        x = tf.nn.relu(x)
        x = tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding="VALID")
        
                
        # 2nd Convolutional Layer
        x = tf.nn.conv2d(x, self.wc2, strides=[1, 1, 1, 1], padding="VALID")
        x = tf.nn.bias_add(x, self.bc2)
        x = tf.nn.relu(x)
        x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")
        
        # Flattten out
        # N X Number of Nodes
        # Flatten()
        x = tf.reshape(x, (tf.shape(x)[0], -1))
        
        # Dense1
        x = tf.matmul(x, self.wd3)
        x = tf.nn.bias_add(x, self.bd3)
        x = tf.nn.relu(x)

        
        # Dense2
        x = tf.matmul(x, self.wd4)
        x = tf.nn.bias_add(x, self.bd4)
        x = tf.nn.relu(x)
        
        
        # Dense3
        x = tf.matmul(x, self.wd5)
        x = tf.nn.bias_add(x, self.bd5)
#         x = tf.nn.sigmoid(x)
        
        return x


def preprocess(image):
    image = tf.expand_dims(image, axis=0)
    image = tf.image.rgb_to_grayscale(image)
    image = tf.image.resize(image, (28, 28))
    image = tf.dtypes.cast(image, tf.float32)
    image = tf.image.per_image_standardization(image)
    return image


def make_predictions(predictions):
    return tf.nn.softmax(predictions, axis=1)


if __name__ == "__main__":
    model = LeNet()

    model_path = './weights.h5'
    print(model_path)
    model.load_weights(model_path)