import tensorflow as tf
import numpy as np
from src.config import TEST_DIR, TRAIN_DIR, WEIGHTS_PATH
from src.preprocess import process_single_image
from src.model import AlexNet

# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

def load_model():
    model = AlexNet()
    model.load_weights(WEIGHTS_PATH)
    return model

def make_predictions(predicted):
    predicted_image_class = tf.nn.softmax(predicted)
    predicted_image_class = tf.argmax(predicted_image_class, axis=1)
    class_name = check_class(predicted_image_class)
    class_name = class_name.numpy().decode('utf-8')
    return class_name

def check_class(pos):
    CLASS_NAMES = np.array([item.name for item in TRAIN_DIR.glob('*')])
    CLASS_NAMES = tf.convert_to_tensor(CLASS_NAMES)
    return CLASS_NAMES[int(pos)]

if __name__ == "__main__":
    model = load_model()
    test_ds = tf.data.Dataset.list_files(str(TEST_DIR / '*/'))
    for img_path in test_ds.take(1):
        print(img_path)
        image = process_single_image(img_path)
        prediction = model(image)
        class_prediction = make_predictions(prediction)
        print(class_prediction)