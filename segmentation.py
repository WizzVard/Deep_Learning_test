import cv2
import argparse
import imutils
import pandas as pd
import numpy as np
from imutils.perspective import four_point_transform

from skimage import transform
from skimage import filters

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='path to the image file')
args = vars(ap.parse_args())

image = cv2.imread(args["image"])

image = imutils.resize(image, height=500)
# perform a blackhat morphological operation that will allow
# us to reveal dark regions (i.e., text) on light backgrounds
# (i.e., the license plate itself)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
rectKern = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKern)
cv2.imshow("Blackhat", blackhat)

def nothing(x):
  pass

cv2.namedWindow("Light Regions")
cv2.createTrackbar("Threshold value", "Light Regions", 66, 255, nothing)




while True:
	# next, find regions in the image that are light
	squareKern = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
	light = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, squareKern)
	value_threshold = cv2.getTrackbarPos("Threshold value", "Light Regions")
	light = cv2.threshold(light, value_threshold, 255, cv2.THRESH_BINARY)[1]

	cv2.imshow("Light Regions", light)
	key = cv2.waitKey(100)
	if key == 27:
		break

# compute the Scharr gradient representation of the blackhat
# image in the x-direction and then scale the result back to
# the range [0, 255]
gradX = cv2.Sobel(blackhat, ddepth=cv2.CV_32F,
	dx=1, dy=0, ksize=-1)
gradX = np.absolute(gradX)
(minVal, maxVal) = (np.min(gradX), np.max(gradX))
gradX = 255 * ((gradX - minVal) / (maxVal - minVal))
gradX = gradX.astype("uint8")
cv2.imshow("Scharr", gradX)

# blur the gradient representation, applying a closing
# operation, and threshold the image using Otsu's method
gradX = cv2.GaussianBlur(gradX, (5, 5), 0)
gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKern)
thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
cv2.imshow("Grad Thresh", thresh)

# perform a series of erosions and dilations to clean up the
# thresholded image
thresh = cv2.erode(thresh, None, iterations=2)
thresh = cv2.dilate(thresh, None, iterations=2)
cv2.imshow("Grad Erode/Dilate", thresh)

cv2.waitKey(0)
