# import the necessary packages
import argparse
import imutils
import cv2
import numpy as np
from imutils import contours
from imutils.perspective import four_point_transform
from skimage import exposure
from skimage.filters import threshold_local
from skimage import transform
from skimage import filters
from PIL import Image
import pytesseract
import os

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help="path to input image to be OCR'd")
ap.add_argument("-p", "--preprocess", type=str, default="thresh", help="type of preprocessing  to be done")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])


cv2.waitKey(0)

