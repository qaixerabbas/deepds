import cv2
import numpy as np
from openvino.runtime import Core
import os
from keras.models import load_model
from typing import List
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.models import Model
import tensorflow as tf

os.environ[
    "TF_CPP_MIN_LOG_LEVEL"
] = "2"  # 0: INFO (default), 1: WARNING, 2: ERROR, 3: FATAL


model_file_name = "v3-small_224_1.0_float.xml"
class_file_name = "imagenet_2012.txt"
model_path = os.path.join(os.getcwd(), "model", model_file_name)
class_path = os.path.join(os.getcwd(), "utils", class_file_name)

### Functions for loading default clases and model


def load_classes_and_model(model_path: str, class_path: str):
    """To load classes and models object from local filesystem

    :param model_path: path to deep learning model on disk
    :type model_path: str
    :param class_path: path to classes file on disk
    :type class_path: str
    :return: compiled model, imagenet classes, and output layer of model
    :rtype: tuple
    """
    ie = Core()
    model = ie.read_model(model=model_path)
    compiled_model = ie.compile_model(model=model, device_name="CPU")
    output_layer = compiled_model.output(0)
    print()
    print("Reading imagenet class data files")
    print()
    imagenet_classes = open(class_path).read().splitlines()
    imagenet_classes = ["background"] + imagenet_classes
    print()
    print("initializing the openvino models")
    print()
    return output_layer, imagenet_classes, compiled_model


def classify_image(image_frame):
    """For classification of (a single) input image

    :param image_frame: Temp image generated during start_streaming function execution
    :type image_frame: str
    :return: Class prediction by OpenVino Model
    :rtype: str
    """
    output_layer, imagenet_classes, compiled_model = load_classes_and_model(
        model_path, class_path
    )
    image = cv2.cvtColor(cv2.imread(image_frame), code=cv2.COLOR_BGR2RGB)
    input_image = cv2.resize(src=image, dsize=(224, 224))
    input_image = np.expand_dims(input_image, 0)
    result_infer = compiled_model([input_image])[output_layer]
    result_index = np.argmax(result_infer)
    name = imagenet_classes[result_index]
    name = name.split(",")[0]
    name = name.split(" ", 1)[1]
    return name


#### functions for loading custom model and classes
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
