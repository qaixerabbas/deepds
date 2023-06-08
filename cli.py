"""
Script Name: deepds cli.py
Author: Qaiser Abbas
Date: June 8, 2023
Email: mqaiser617@gmail.com
"""

from vidgear.gears import VideoGear
from vidgear.gears import CamGear
import cv2
import argparse
import os
import sys
import time
from utils.functions import classify_image
from utils.functions import classify_keras_style

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

parser.add_argument(
    "--custom_model", required=False, type=str, help="Path to your custom model."
)

parser.add_argument(
    "--labels", required=False, type=str, help="Path to your labels file."
)


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

        if args.custom_model and args.labels:
            print("Using Custom Model and Labels...")
            class_name = classify_keras_style(args.custom_model, temp_img, args.labels)
        else:
            print("Using default model and labels...")
            class_name = classify_image(temp_img)

        os.remove(temp_img)

        if not os.path.exists(default_path + str(class_name)):
            os.makedirs(default_path + class_name)
            os.chdir(default_path + class_name)
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
    time_start = time.time()
    args = parser.parse_args()
    video_path = args.video_path
    allowed_video_format = [".mp4", ".ogg", ".mkv", ".mov", ".avi", ".webm"]
    try:
        if args.video_path[-4:] in allowed_video_format:
            print(f"Found a valid video file with {args.video_path[-4:]} format.")
            print("Processing current video file...")
            stream_object = VideoGear(source=video_path).start()
            start_streaming(stream_object)
        elif args.video_path[:4] == "http" and "youtube.com" in args.video_path:
            print("File found with youtube video link")
            print("Processing current YouTube link...")
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
    time_end = time.time()
    total_time = time_end - time_start
    print(f"Total time taken to generate dataset: {total_time} seconds")
    sys.exit()
