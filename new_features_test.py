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
    description="Python program to auto generate datasets for computer vision applications.",
    epilog="That's how create datasets Simple and Nice",
)

parser.add_argument(
    "--video_path", required=True, type=str, help="Target video link to YouTube video"
)

parser.add_argument(
    "--destination", required=True, type=str, help="Target destination to save dataset"
)


def classify_image(image_frame):
    output_layer, imagenet_classes, compiled_model = load_classes_and_model(
        model_path, class_path
    )
    image = cv2.cvtColor(cv2.imread(image_frame), code=cv2.COLOR_BGR2RGB)
    input_image = cv2.resize(src=image, dsize=(224, 224))
    input_image = np.expand_dims(input_image, 0)
    result_infer = compiled_model([input_image])[output_layer]
    result_index = np.argmax(result_infer)
    name = imagenet_classes[result_index]
    name = name.split()[1]
    return name


args = parser.parse_args()

video_path = args.video_path


def start_streaming(stream_object):
    default_path = (
        args.destination
    )  # "D:\\github codes\\test2\\" ==> \\ must be used with \\ in the end as well
    stream = stream_object
    currentframe = 0
    while True:

        frame = stream.read()  # using functions from vidGear module
        if frame is None:
            break

        cv2.imshow("Output Frame", frame)  # optional if u want to show the frames
        temp_img = "temp.png"
        cv2.imwrite(temp_img, frame)
        print(f"Classifying image {temp_img}")
        class_name = classify_image(temp_img)
        print(
            f"#############==> Predicted Class Name: {class_name} <==#########################"
        )
        print(f"Removing temporary image {class_name}")
        os.remove(temp_img)

        if not os.path.exists(default_path + str(class_name)):
            print(f"Making dir {class_name}")
            os.makedirs(default_path + class_name)
            # path = path + str(class_name)
            print(f"Entering into {default_path + class_name}")
            # default_path=default_path+str(class_name)
            os.chdir(default_path + class_name)
            print(f"Writing frames to dir {default_path + class_name}")
            name = str(class_name) + str(currentframe) + ".jpg"  # test/labrador/0.jpg
            cv2.imwrite(name, frame)
        elif os.path.exists(default_path + str(class_name)):
            # default_path=default_path+str(class_name)
            os.chdir(default_path + class_name)
            name = str(class_name) + str(currentframe) + ".jpg"
            cv2.imwrite(name, frame)
        else:
            pass

        # cv2.imwrite(name, frame)
        currentframe += 30  # chnage 5 with the number of frames. Here 5 means capture frame after every 5 frames
        # usually videos are 30fps so if here 30 is provided a frame will be captures after every second.

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    cv2.destroyAllWindows()
    stream.stop()


def load_classes_and_model(model_path, class_path):
    ie = Core()
    model = ie.read_model(model=model_path)
    compiled_model = ie.compile_model(model=model, device_name="CPU")
    output_layer = compiled_model.output(0)
    print("Reading imagenet class data files")
    imagenet_classes = open(class_path).read().splitlines()
    imagenet_classes = ["background"] + imagenet_classes
    print("initializing the openvino models")
    return output_layer, imagenet_classes, compiled_model


model_path = "D:\\github codes\\deepds\\model\\v3-small_224_1.0_float.xml"
class_path = "D:\\github codes\\deepds\\utils\\imagenet_2012.txt"

if __name__ == "__main__":
    # from sys import argv
    allowed_video_format = [".mp4", ".ogg", ".mkv", ".mov", ".avi", ".webm"]
    try:
        if args.video_path[-4:] in allowed_video_format:
            print(f"Found valid video file with {args.video_path[-4:]} format.")
            # load_classes_and_model()
            stream_object = VideoGear(source=video_path).start()
            # local_video_file(video_path)
            start_streaming(stream_object)
        elif args.video_path[:4] == "http" and "youtube.com" in args.video_path:
            print("File found with youtube video link")
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
