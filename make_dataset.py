import cv2
# import shutil
# need to add a function for local video dataset and local model maker and update argparase
# above task can be done using vidgear so no need for OpenCV's dedicated function
# reference => https://abhitronix.github.io/vidgear/latest/gears/videogear/usage/
import os
from vidgear.gears import CamGear
import argparse
import sys
# import requests
# import re
# import json
import matplotlib.pyplot as plt
import numpy as np
from openvino.runtime import Core

parser = argparse.ArgumentParser(prog = "make-dataset", description="Program to auto make datasets.", epilog="That's how create datasets Simple and Nice")

parser.add_argument("--videolink", required=True, type=str, help="Target video link to YouTube video")

parser.add_argument(
    "--destination", required=True, type=str, help="Target destination to save dataset"
)

args = parser.parse_args()

# https://www.youtube.com/watch?v=oQyKL_jBz0Q&ab_channel=MLTNA7X
print("Starting to read stream...")
stream = CamGear(
    source=args.videolink,
    stream_mode=True,
    time_delay=1,
    logging=True,
).start()

print("Reading imagenet class data files")
imagenet_classes = open("utils/imagenet_2012.txt").read().splitlines()
imagenet_classes = ['background'] + imagenet_classes

print("initializing the openvino models")
ie = Core()
model = ie.read_model(model="model/v3-small_224_1.0_float.xml")
compiled_model = ie.compile_model(model=model, device_name="CPU")
output_layer = compiled_model.output(0)

def classify_image(image_frame):
    image = cv2.cvtColor(cv2.imread(image_frame), code=cv2.COLOR_BGR2RGB)
    input_image = cv2.resize(src=image, dsize=(224, 224))
    input_image = np.expand_dims(input_image, 0)
    result_infer = compiled_model([input_image])[output_layer]
    result_index = np.argmax(result_infer)
    name = imagenet_classes[result_index]
    name = name.split()[1]
    return name

def main():
    default_path = args.destination # "D:\\github codes\\test2\\" ==> \\ must be used with \\ in the end as well

    currentframe = 0
    while True:

        frame = stream.read()  ### using functions from vidGear module
        if frame is None:
            break

        # cv2.imshow("Output Frame", frame)  # optional if u want to show the frames

        # name = path + str(currentframe) + ".jpg" # test/0.jpg
        # print("Reading..." + name)
        temp_img = 'temp.png'
        cv2.imwrite(temp_img, frame)
        print(f"Classifying image {temp_img}")
        class_name = classify_image(temp_img)
        # print(type(class_name))
        print(f"#############==> Predicted Class Name: {class_name} <==#########################")
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
            name = str(class_name) + str(currentframe) + ".jpg" # test/labrador/0.jpg
            cv2.imwrite(name, frame)
        elif os.path.exists(default_path + str(class_name)):
            # default_path=default_path+str(class_name)
            os.chdir(default_path + class_name)
            name =  str(class_name) + str(currentframe) + ".jpg"
            cv2.imwrite(name, frame)
        else:
            pass

        # cv2.imwrite(name, frame)
        currentframe += 30  ##chnage 5 with the number of frames. Here 5 means capture frame after every 5 frames
        ###usually videos are 30fps so if here 30 is provided a frame will be captures after every second.

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    cv2.destroyAllWindows()
    stream.stop()

if __name__ == '__main__':
    # from sys import argv
    try:
        main()
    except KeyboardInterrupt:
        pass
    sys.exit()
