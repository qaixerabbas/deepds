"""
Script Name: DeepDS cli.py
Author: Qaiser Abbas
Date: May 17, 2023
Email: mqaiser617@gmail.com
"""

from vidgear.gears import VideoGear
from vidgear.gears import CamGear
import cv2
import argparse
import numpy as np
from openvino.runtime import Core
import os
import sys

parser = argparse.ArgumentParser(
    prog="DeepDS",
    description="Python script to auto generate datasets for computer vision applications.",
    epilog="Powered by deeplearning",
)

parser.add_argument(
    "--video_path",
    required=True,
    type=str,
    help="Target video link to YouTube video or local video path.",
)

parser.add_argument(
    "--destination", required=True, type=str, help="Target destination to save dataset."
)

parser.add_argument(
    "--displayframe",
    required=False,
    action="store_true",
    help="Display the frames currently being processed.",
)

model_file_name = "v3-small_224_1.0_float.xml"
class_file_name = "imagenet_2012.txt"
model_path = os.path.join(os.getcwd(), "model", model_file_name)
class_path = os.path.join(os.getcwd(), "utils", class_file_name)


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


def start_streaming(stream_object):
    """To start streaming and main logic of the script
    The function takes a stream object and then captures frames, classifies the intermediate frames and sorts them into classified respective classes based directories.

    :param stream_object: VidGear or CamGear stream object to start streaming
    :type stream_object:  instance of CamGear or VidGear class
    """
    default_path = (
        args.destination
    )  # "D:\\github codes\\test2\\" ==> \\ must be used with \\ in the end as well
    show_frame = args.displayframe
    stream = stream_object
    currentframe = 0
    while True:
        frame = stream.read()  # using functions from vidGear module
        if frame is None:
            break
        if show_frame:
            cv2.imshow(
                "Current Output Frame", frame
            )  # optional if u want to show the frames
        temp_img = "temp.png"
        cv2.imwrite(temp_img, frame)
        print()
        print(f"Classifying image {temp_img}")
        print()
        class_name = classify_image(temp_img)
        print()
        print(
            f"#############==> Predicted Class Name: {class_name} <==#########################"
        )
        print()
        print(f"Removing temporary image {class_name}")
        print()
        os.remove(temp_img)

        if not os.path.exists(default_path + str(class_name)):
            print()
            print(f"Creating {class_name} directory.")
            print()
            os.makedirs(default_path + class_name)
            print()
            print(f"Navigating into {default_path + class_name}")
            print()
            os.chdir(default_path + class_name)
            print()
            print(f"Writing frames to {default_path + class_name}")
            print()
            name = str(class_name) + str(currentframe) + ".jpg"
            cv2.imwrite(name, frame)
        elif os.path.exists(default_path + str(class_name)):
            os.chdir(default_path + class_name)
            name = str(class_name) + str(currentframe) + ".jpg"
            cv2.imwrite(name, frame)
        else:
            pass

        currentframe += 1

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    cv2.destroyAllWindows()
    stream.stop()


if __name__ == "__main__":
    args = parser.parse_args()
    video_path = args.video_path
    allowed_video_format = [".mp4", ".ogg", ".mkv", ".mov", ".avi", ".webm"]
    try:
        if args.video_path[-4:] in allowed_video_format:
            print()
            print(f"Found valid video file with {args.video_path[-4:]} format.")
            stream_object = VideoGear(source=video_path).start()
            print()
            start_streaming(stream_object)
        elif args.video_path[:4] == "http" and "youtube.com" in args.video_path:
            print()
            print("File found with youtube video link")
            print()
            stream_object = CamGear(
                source=video_path,
                stream_mode=True,
                time_delay=1,
                logging=True,
            ).start()
            start_streaming(stream_object)
        else:
            raise ValueError(
                "Please provide a valid video file path or youtube video link."
            )
    except KeyboardInterrupt:
        pass
    sys.exit()
