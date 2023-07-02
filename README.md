# object-detection-using-python
This repository contains code for performing object detection using a pre-trained model in OpenCV. The code is written in Python and demonstrates how to detect objects in both images and real-time video streams.

## Dependencies
Make sure you have the following dependencies installed:
- OpenCV
- matplotlib

## Setup
Clone the repository: <br/>
git clone 'repository-url'
<br/>
Download the pre-trained model and configuration file.

- Download the frozen_inference_graph.pb file from <model-url> and place it in the repository directory.
- Download the ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt configuration file from <config-url> and place it in the repository directory.
<br/>
Create a virtual environment (optional but recommended): <br/>
python3 -m venv env <br/>
source env/bin/activate
<br/>
Install the required dependencies

## Usage
### Image Object Detection
1. Place the image you want to perform object detection on in the repository directory.
2. Run the following command to detect objects in the image:
<br/>
python detect_image.py --image <image-path>

<br/>
The program will display the image with bounding boxes and labels around the detected objects.

### Real-Time Video Object Detection
1. Connect a webcam or place a video file in the repository directory.
2. Run the following command to perform real-time object detection on the video feed:
<br/>
python detect_video.py --video <video-path>
<br/>
The program will open a window showing the live video feed with bounding boxes and labels around the detected objects.


