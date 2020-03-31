import base64
import json
import os
import numpy as np
import requests

import tensorflow as tf

from PIL import Image
from io import BytesIO
from flask import Flask, request, jsonify

from model import LeNet, preprocess, make_predictions

from keras.applications.imagenet_utils import preprocess_input, decode_predictions


BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
print(BASE_DIR)

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


app = Flask(__name__)


def load_model():
    model_path = './weights.h5'
    model = LeNet()
    model.load_weights(model_path)
    return model


def read_image(files):
    # Files are bytes
    image = Image.open(files)
    image = np.array(image)
    return image

@app.route('/hello', methods=['GET'])
def hello_world():
    return 'Hello, world'


@app.route('/predict',methods=['POST'])
def predict():
    model = load_model()
    files = request.files['images']
    image = read_image(files)
    image = preprocess(image)


    # response = np.array_str(np.argmax(out,axis=1))
    predicted = model(image)

    value = make_predictions(predicted)

    maxArg = tf.argmax(value,axis=1)
    print(maxArg)
    maxarg = maxArg.numpy()

    
    print("------------------")
    print(value)
    print("------------------")
    response = {
        'target': np.array_str(value.numpy()),
        'Result': f"The dress is {class_names[maxarg[0]]}."
    }

    return response


if __name__ == "__main__":
    app.run()