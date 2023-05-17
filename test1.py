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
orig = image.copy()
#Filter
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
bfilter = cv2.bilateralFilter(gray, 13, 17, 17)#Noise reduction
edged_1 = cv2.Canny(bfilter, 30, 200)

#Find contours
cnts = cv2.findContours(edged_1.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

contours,hierarchy =cv2.findContours(edged_1,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

# Loop through contours and find the two biggest area.
for cont in contours:
    area=cv2.contourArea(cont)
    if area>150:
        #print(area)
        cv2.drawContours(image,cont,-1,(0,0,255),5)

# Save your pictures with the contour in red
cv2.imshow('Image with planes in Red.jpg',image)

cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]

location = None
for contour in cnts:
    approx = cv2.approxPolyDP(contour, 10, True)
    if len(approx) == 4:
        location = approx
        break

warped = four_point_transform(gray, location.reshape(4, 2))
output = four_point_transform(image, location.reshape(4, 2))


cv2.rectangle(image, tuple(approx[0][0]), tuple(approx[2][0]), (0, 255, 0), 3)

cv2.imshow("Scanned", imutils.resize(warped, width = 450))
cv2.imshow("Edged", edged_1)
cv2.imshow("Image", image)



cv2.waitKey(0)