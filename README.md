<p align="center">
  <img src="./images/logo2.png">
</p>

# What's DeepDS
One command data creation for visual information processing (weakly/self-supervised labels)

Make your life easy :) 

## Build datasets for computer vision

You can use this script to create datasets with weak image annotations.

### Why Weak Annotations?
Labeling images is a time consuming process. In computer vision, classification labels are the easiest one to annotate input data for ML models. While bounding boxes (detection/localization) and segmentation (semantic/instance) are hard and tedious plus time consuming task. 

Capturing individual images is a lot more time consuming process than capturing videos. This scripts accepts a video (local or YouTube video support) and allows to develop a dataset in ImageNet style annotations. Where each frame in the video is classified using a OpenVino optiimzed model (currently supports InceptionV3). Based on this information individual folders are created and associated images are transferred to the corresponding folders.

Once you have dataset, you can simply annotate these images for Detection/Segmentation tasks. For classification, custom models are required, but using a ImageNet model might help in weak labels. The generated dataset can be used in Unsupervised/Self-supervised settings to pre-train a vision model. After that a small amount of labeled data is enuogh to train robust vision models.

### Prerequisites
Before running the script, make sure you have the installed dependencies using the following command:

```
pip install -r requirements.txt
```

### Step 1. pip install -r requirements.txt
Then select the output directory where you want to save the frames (edit python code to set paths and directories)

### Step 2. update the .py file 
Add youtube video and output dir in python script

### step 3. python <script-name.py>

It will start capturing frames from youtube video and download it into the specified folders.



Usage
---
Alternatively you can use the cli.py file if you do not want to manually modify the main script.
```
>>> python cli.py --help
>>> usage: downlaod_youtube_frames. [-h] --videolink VIDEOLINK --destination DESTINATION

  Program to automatically download youtube images datasets.

  optional arguments:
    -h, --help            show this help message and exit
    --videolink VIDEOLINK
                         YouTube video link
    --destination DESTINATION
                         Target path to save imgz

  A simple and nice cli script to create youtube datasets
```

How to run on cli/terminal?

Example run

``` >>> python cli.py --videolink "https://www.youtube.com/watch?v=PWRg_wak9oI" --destination "D:\dataset\test" ```

Running like above example will start downloading the video frames into the provided destination. Optionally, you can provide ``` --showframe ``` argument in command line if you want to show the frames that are being saved to local disk.

Example

``` >>> python cli.py --videolink "https://www.youtube.com/watch?v=PWRg_wak9oI" --destination "D:\dataset\test" --showframe ```

NOTE: Make sure ``` --videolink ``` and ``` --destination ``` are both strings (enclosed in quotation marks)

### Todo

- [ ] Add framecount in argparse 
- [ ] work with any online videos ( YouTube + more )

### In Progress

- [ ] Working on improving readme, How-To and upload on PyPI. 

### Done ✓

- [x] Capturing frames from remote YouTube videos.
- [x] merge this to deepds
- [x] Saving frames to target directory.
- [x] Cli.py file for easy usage.
