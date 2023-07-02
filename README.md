# object-detection-using-python

The provided code demonstrates how to perform object detection using a pre-trained model in OpenCV. Here's a breakdown of the code:

Import the necessary libraries:
import cv2
import matplotlib.pylab as plt

Set the paths for the configuration file and frozen model:
config_file = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model = 'frozen_inference_graph.pb'

Create an instance of the cv2.dnn_DetectionModel class:
model = cv2.dnn_DetectionModel(frozen_model, config_file)

Load the class labels from a file:
classLabels = []
filename = 'labels.txt'
with open(filename, 'rt') as f:
    classLabels = f.read().rstrip('\n').split('\n')

Set the input parameters for the model:
model.setInputSize(320, 320)
model.setInputScale(1.0 / 127.5)
model.setInputMean((127.5, 127.5, 127.5))
model.setInputSwapRB(True)

Load an image and perform object detection on it:
img = cv2.imread('boy.jpg')
classIndex, confidence, bbox = model.detect(img, confThreshold=0.5)

Visualize the detected objects in the image:
font_scale = 3
font = cv2.FONT_HERSHEY_PLAIN
for classInd, conf, boxes in zip(classIndex.flatten(), confidence.flatten(), bbox):
    cv2.rectangle(img, boxes, (255, 0, 0), 2)
    cv2.putText(img, classLabels[classInd - 1], (boxes[0] + 10, boxes[1] + 40), font, fontScale=font_scale,
                color=(0, 255, 0), thickness=3)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()

Open a video or webcam feed and perform object detection in real-time:
cap = cv2.VideoCapture('video1.mp4')  # video
cap = cv2.VideoCapture(1)  # webcam
if not cap.isOpened():
    cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Can't open the video")

while True:
    ret, frame = cap.read()
    classIndex, confidence, bbox = model.detect(frame, confThreshold=0.65)
    if len(classIndex) != 0:
        for classInd, conf, boxes in zip(classIndex.flatten(), confidence.flatten(), bbox):
            if classInd <= 80:
                cv2.rectangle(frame, boxes, (255, 0, 0), 2)
                cv2.putText(frame, classLabels[classInd - 1], (boxes[0] + 10, boxes[1] + 40), font,
                            fontScale=font_scale, color=(0, 255, 0), thickness=3)
    cv2.imshow('Object Detection', frame)
    if cv2.waitKey(2) & 0xff == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

This code demonstrates object detection on both an image and video feed using the pre-trained model specified by the configuration file and frozen model. Detected objects are highlighted with bounding boxes and labeled with class names.
