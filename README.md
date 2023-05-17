<p align="center">
  <img src="./images/logo2.png"> <br>
  Version: 0.1.0
</p>


# What's DeepDS
One command data creation for visual information processing (weakly/self-supervised settings)

Make your life easy :) 

## Build datasets for computer vision

You can use this script to create datasets with weak image annotations.

## Why Weak Annotations?
Labeling images is a time consuming process. In computer vision, classification labels are the easiest one to annotate input data for ML models. While bounding boxes (detection/localization) and segmentation (semantic/instance) are hard and tedious plus time consuming task. 

Capturing individual images is a lot more time consuming process than capturing videos. This scripts accepts a video (local or YouTube video support) and allows to develop a dataset in ImageNet style annotations. Where each frame in the video is classified using a OpenVino optiimzed model (currently supports InceptionV3). Based on this information individual folders are created and associated images are transferred to the corresponding folders.

Once you have dataset, you can simply annotate these images for Detection/Segmentation tasks. For classification, custom models are required, but using a ImageNet model might help in weak labels. The generated dataset can be used in Unsupervised/Self-supervised settings to pre-train a vision model. After that a small amount of labeled data is enuogh to train robust vision models.

The current InceptionV3 model is optimized for CPU so it's performance is not comparable with SOTA classification models. However, the predictions from this model are good enough to be used in weakly supervised training.

## Prerequisites
**Recommended** Create a new virtual environment. For creating a new environment follow the instructions below:

To create and activate a virtual environment in Windows, follow these steps:

1. Open a command prompt:
2. Press the Windows key and type "cmd".
3. Press Enter and create a virtual environment using following command:
```
python -m venv myenv
```
Replace "myenv" with the desired name for your virtual environment.

4. Activate the virtual environment:
```
myenv\Scripts\activate
```
Replace "myenv" with the name you provided in the previous step.

5. After executing these commands, your virtual environment will be created and activated. Once the virtual environment is ready just run the following command:

```
pip install -r requirements.txt
```

Usage
---
Use cli.py file if you do not want to manually modify the main script.
```
>>> python cli.py --help
>>> usage: DeepDS [-h] --video_path VIDEO_PATH --destination DESTINATION [--displayframe]

Python script to auto generate datasets for computer vision applications.

optional arguments:
  -h, --help            show this help message and exit
  --video_path VIDEO_PATH
                        Target video link to YouTube video or local video path.
  --destination DESTINATION
                        Target destination to save dataset.
  --displayframe        Display the frames currently being processed.

Powered by deeplearning
```

Example run
---

```
 >>> python cli.py --video_path "https://www.youtube.com/watch?v=PWRg_wak9oI" --destination "D:\\dataset\\test\\"
```

Optionally, you can provide ``` --displayframe ``` argument in command line if you want to display current frames.

``` 
>>> python cli.py --video_path "https://www.youtube.com/watch?v=PWRg_wak9oI" --destination "D:\\dataset\\test" --displayframe 
```

NOTE: Make sure ``` --video_path ``` and ``` --destination ``` are both strings (enclosed in quotation marks)

### Todo

- [ ] Add a flag for skipping frames in video streams
- [ ] Add custom model support 
- [ ] Work with any online videos
- [ ] Add train_test_split function
- [ ] Add support for loading open source datasets

### In Progress

- [ ] Working on improving readme and uploading to PyPI. 

### Done ✓

- [x] Capture frames from local & YouTube videos without downloading.
- [x] Create directories based on class predictions by deep learning model.
- [x] Arrange the images into a proper imageNet based annotations
- [x] Add a showframe argument for displaying current frames
