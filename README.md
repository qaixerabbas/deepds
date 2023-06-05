<h1 align="center">
  <img src="./images/deepds.svg" height="200" width="200"> <br>
  <b style="font-size: 26px">DeepDS<b> <br>
</h1>
<!--     <p align="center"> Version: 0.1.0 </p> -->
    
<p align="center">
Grab your coffee and let me analyze the data :)
</p>
<!-- # What's DeepDS
One command data creation for visual information processing (weakly/self-supervised settings) -->
&nbsp;

DeepDS is an **automatic Video Processing Python** tool that provides an easy-to-use command line interface to autoamte your data creation process for various machine learning tasks. It uses **OpenCV** and **VidGear** at the backend and uses **Optimized Deep Learning** algorithm to analyze and create your datasets with just a single command. 

DeepDs primarily focuses on automation of repetetive tasks, and thereby lets ML practitionars & researchers/developers to easily create image datasets from online/offline videos. Additionally you can use your own trained ML models for custom dataset creation tasks.
    
The output of the data creation project will be directories containing images that are ready to be used for finetuning/training your ML models
   
    
    - dataset
      - Class 1
        - img_01
        - img_02
        .
        - img_n
    
      - Class 2
        - img_01
        - img_02
        .
        - img_n
    
      - Class N
        - img_01
        - img_02
        .
        - img_n
    
&nbsp;    
    
<!-- ## Build datasets for computer vision

You can use this script to create datasets with weak image annotations. -->

## Why Weak Annotations?
Labeling images is a time consuming process. In computer vision, classification labels are the easiest one to annotate input data for ML models. While bounding boxes (detection/localization) and segmentation (semantic/instance) are hard and tedious plus time consuming task. 

Capturing individual images is a lot more time consuming process than capturing videos. This scripts accepts a video (local or YouTube video support) and allows to develop a dataset in ImageNet style annotations. Where each frame in the video is classified using a OpenVino optiimzed model (currently supports InceptionV3). Based on this information individual folders are created and associated images are transferred to the corresponding folders.

Once you have dataset, you can simply annotate these images for Detection/Segmentation tasks. For classification, custom models are required, but using a ImageNet model might help in weak labels. The generated dataset can be used in Unsupervised/Self-supervised settings to pre-train a vision model. After that a small amount of labeled data is enuogh to train robust vision models.

The current InceptionV3 model is optimized for CPU so it's performance is not comparable with SOTA classification models. However, the predictions from this model are good enough to be used in weakly supervised training.

## Prerequisites

Create a new virtual environment. For creating a new environment follow the instructions below:

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
6. For [MacOS](https://programwithus.com/learn/python/pip-virtualenv-mac) and [Linux](https://www.geeksforgeeks.org/creating-python-virtual-environment-windows-linux/) please follow the instructions for creating & activating the virtual environment.

Usage
---
Use cli.py file if you do not want to manually modify the main script.
```
>>> python cli.py --help
>>> usage: DeepDS [-h] --video_path VIDEO_PATH --destination DESTINATION [--displayframe] [--custom_model CUSTOM_MODEL]
              [--labels LABELS]

Python script to auto generate datasets for computer vision applications.

optional arguments:
  -h, --help            show this help message and exit
  --video_path VIDEO_PATH	    ||  Target video link to YouTube video or local video path.
  --destination DESTINATION  	    ||  Target destination to save dataset.
  --displayframe             	    ||  Display the frames currently being processed.
  --custom_model CUSTOM_MODEL       ||  Path to your custom model.
  --labels LABELS       	    ||  Path to your labels file.

Powered by deeplearning
```

Example run
---

```
 >>> python cli.py --video_path "https://www.youtube.com/watch?v=ffCWuQhj9Fo" --destination "D:\\dataset\\test2\\" --custom_model "D:\\dataset\\bees_keras_model.h5" --labels "D:\\dataset\\labels.txt"
```
If you do not provide trained model and labels path, the script will automatically load ImageNet labels and pre-trained Intel OpenVino optimized Inception model for classification.
Optionally, you can provide ``` --displayframe ``` argument in command line if you want to display current frames.

``` 
>>> python cli.py --video_path "https://www.youtube.com/watch?v=ffCWuQhj9Fo" --destination "D:\\dataset\\test2\\" --custom_model "D:\\dataset\\bees_keras_model.h5" --labels "D:\\dataset\\labels.txt" --displayframe
```

NOTE: Make sure ``` --video_path ```, ``` --destination ```, ``` --custom_model ```, ``` --labels ``` are strings (enclosed in quotation marks)

### Todo

- [ ] Add a flag for skipping frames in video streams
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
- [x] Add custom model support 

### LIcense
This project is licensed under the [MIT License](https://github.com/qaixerabbas/deepds/blob/master/LICENSE).

### Acknowledgments
This script was developed with the help of various open-source libraries and resources. I would like to acknowledge their contributions to the project:
- OpenCV: https://opencv.org/
- VidGear: https://github.com/abhiTronix/vidgear
- OpenVino: https://github.com/openvinotoolkit/openvino
- PyTorch: https://github.com/pytorch/pytorch

### Limitations
1. Currently, It only works with the OpenVino optimized model (.xml) files.
2. It only works with images from ImageNet dataset. Any image category that isn't available in ImageNet will be ignored and randomly(upto some threshold) assigned image class.
    
### Implementation Ideas
``` FPS = 30
Len(Video) = 5 mins
Total frames = 5 * 60 * 30 => 9000 
train = 70% of total frames => 6300
test = 30% of total frames => 2700
    for frame in frames:
	if num(frame)<=6300
	  move to train
	else:
	  move to test
karpathy.github.io/2015/11/14/ai/ ```
