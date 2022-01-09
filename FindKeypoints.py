import matplotlib.pyplot as plt
import numpy as np
import cv2
import keras
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import os
from PIL import Image
from skimage import transform
from keras.models import load_model


# The input image.
image = cv2.imread('056.png')

grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Set up the SimpleBlobdetector with default parameters.
params = cv2.SimpleBlobDetector_Params()

# Define thresholds
#Can define thresholdStep. See documentation.
params.minThreshold = 0
params.maxThreshold = 180

# Filter by Area.
params.filterByArea = True
params.minArea = 0
params.maxArea = 1000

# Filter by Color (black=0)
params.filterByColor = False
params.blobColor = 1

# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0
params.maxCircularity = 2

# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0
params.maxConvexity = 2

# Filter by InertiaRatio
params.filterByInertia = True
params.minInertiaRatio = 0
params.maxInertiaRatio = 1

# Distance Between Blobs
params.minDistBetweenBlobs = 1

# Setup the detector with parameters
detector = cv2.SimpleBlobDetector_create(params)

# Detect blobs
keypoints = detector.detect(grayImage)

print("Number of blobs detected : ", len(keypoints))
#print(keypoints[0].pt)

# Draw blobs
img_with_blobs = cv2.drawKeypoints(grayImage, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Save result
#cv2.imwrite("./output/1.png", img_with_blobs)

#####################################################
i = 0
i = 0
j = 0
thickness = 2
color = (0,0,255)
boxes =[]

for i in range(0,len(keypoints)):
    x , y = keypoints[i].pt
    ROI = grayImage[int(y)-30:int(y)+30, int(x)-30:int(x)+30]
    
    coordinate = (int(x)-30, int(y)-30, int(x)+30, int(y)+30)
    boxes.append(coordinate)
    for j in range(len(boxes)):
        image = cv2.rectangle(image, (boxes[i][0]+10,boxes[i][1]+10), (boxes[i][2]+10,boxes[i][3]+10), color, thickness)
        j += 1
    cv2.imwrite("./output/12.png", image)

    i += 1

#print(boxes)
