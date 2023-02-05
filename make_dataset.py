import cv2
import os
from vidgear.gears import CamGear
import argparse
import sys
import requests
import re
import json
import matplotlib.pyplot as plt
import numpy as np
from openvino.runtime import Core

stream = CamGear(
    source="https://www.youtube.com/watch?v=oQyKL_jBz0Q&ab_channel=MLTNA7X",
    stream_mode=True,
    time_delay=1,
    logging=True,
).start()

imagenet_classes = open("utils/imagenet_2012.txt").read().splitlines()
imagenet_classes = ['background'] + imagenet_classes

ie = Core()
model = ie.read_model(model="model/v3-small_224_1.0_float.xml")
compiled_model = ie.compile_model(model=model, device_name="CPU")
output_layer = compiled_model.output(0)

def classify_image(image):
    image = cv2.cvtColor(cv2.imread(image), code=cv2.COLOR_BGR2RGB)
    input_image = cv2.resize(src=image, dsize=(224, 224))
    input_image = np.expand_dims(input_image, 0)
    result_infer = compiled_model([input_image])[output_layer]
    result_index = np.argmax(result_infer)
    name = imagenet_classes[result_index]
    return name

def main():
    path = "D:\\github codes\\test\\"

    currentframe = 0
    while True:

        frame = stream.read()  ### using functions from vidGear module
        if frame is None:
            break

        # cv2.imshow("Output Frame", frame)  # optional if u want to show the frames

        name = path + "./frames" + str(currentframe) + ".jpg"
        print("classifying..." + name)
        class_name = classify_image(name)
        
        if not os.path.exists(class_name):
            os.makedirs(class_name)


        cv2.imwrite(name, frame)
        currentframe += 30  ##chnage 5 with the number of frames. Here 5 means capture frame after every 5 frames
        ###usually videos are 30fps so if here 30 is provided a frame will be captures after every second.

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    cv2.destroyAllWindows()
    stream.stop()

if __name__ == '__main__':
    from sys import argv
    try:
        main(argv)
    except KeyboardInterrupt:
        pass
    sys.exit()