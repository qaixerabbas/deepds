from keras.models import load_model
import numpy as np
from typing import List
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.models import Model
import tensorflow as tf
import os

# Set the TensorFlow logging level to suppress messages
os.environ[
    "TF_CPP_MIN_LOG_LEVEL"
] = "2"  # 0: INFO (default), 1: WARNING, 2: ERROR, 3: FATAL


def load_keras_model(model_path: str) -> Model:
    """loads a pre-trained keras model

    :param model_path: path to your saved and trained keras model
    :type model_path: str
    :return: a trained keras model
    :rtype: Model
    """
    model = load_model(model_path, compile=False)
    return model


def load_custom_labels(label_file_path: str) -> List[str]:
    """loads custom labels from a .txt file

    :param label_file_path: path to your .txt file containing your labels
    :type label_file_path: str
    :raises TypeError: If file is not found or unable to read the file
    :return: a list of class labels
    :rtype: List[str]
    """
    if type(label_file_path) == str:
        try:
            class_labels = open(label_file_path, "r").readlines()
            return class_labels
        except FileNotFoundError:
            print("Error: File not found.")
        except IOError:
            print("Error: Unable to read the file.")
        except Exception as e:
            print("An unexpected error occurred:", str(e))
    else:
        raise TypeError("Labels file path must be a string")


def classify_keras_style(model_path: str, image_path: str, labels_path: str) -> str:
    """loads a keras model and classifies an image based on trained model

    :param model_path: path to a trained keras model
    :type model_path: str
    :param image_path: path to your image to be classified
    :type image_path: str
    :param labels_path: path to a .txt file containing labels
    :type labels_path: str
    :return: a string of the class name as predicted by the model
    :rtype: str
    """
    model = load_keras_model(model_path)
    labels = load_custom_labels(labels_path)
    img = load_img(image_path, target_size=(224, 224))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x)
    index = np.argmax(preds)
    class_name = labels[index]
    label_list = class_name.split(" ")
    class_name = " ".join(label_list[1:])
    return class_name


model_path = "bees_keras_Model.h5"
img_path = "test/ants/ants (4).jpg"

res = classify_keras_style(model_path, img_path, "labels.txt")
print(res)
