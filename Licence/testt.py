import cv2
import argparse
import imutils
import pandas as pd
import numpy as np
import pytesseract
from imutils.perspective import four_point_transform
import easyocr
import re

from skimage import transform
from skimage import filters

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='path to the image file')
args = vars(ap.parse_args())

image = cv2.imread(args["image"])

image = imutils.resize(image, height=500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
bfilter = cv2.bilateralFilter(gray, 13, 17, 17)
edged_1 = cv2.Canny(bfilter, 30, 200)


cnts = cv2.findContours(edged_1.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]

location = None
for contour in cnts:
    approx = cv2.approxPolyDP(contour, 10, True)
    if len(approx) == 4:
        location = approx
        break

warped = imutils.resize(four_point_transform(gray, location.reshape(4, 2)), width=450)

reader = easyocr.Reader(['en'])
detection = reader.readtext(warped)

if len(detection) == 0:
    text = "Impossible to detect text"
    cv2.putText(image, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 3)

else:
    blur = cv2.GaussianBlur(warped, (3, 3), 0)
    ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

    # create rectangular kernel for dilation
    rect_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    # apply dilation to make regions more clear
    erode = cv2.erode(thresh, rect_kern, iterations=1)
    dilation = cv2.dilate(erode, rect_kern, iterations=1)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    sorted_contours = sorted(contours, key=lambda cntr: cv2.boundingRect(cntr)[0])

    # Create a blank image to draw contours on
    contour_image = np.zeros_like(erode)

    # Draw contours on the blank image
    cv2.drawContours(contour_image, sorted_contours, -1, (255, 255, 255), 2)

    # Display the image with contours
    cv2.imshow("Contours", contour_image)
    plate_num = ""

    for cnt in sorted_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        height, width = dilation.shape
        if height / float(h) > 20: continue

        ratio = h / float(w)
        if ratio < 1 or ratio > 3: continue

        if width / float(w) > 24: continue

        area = h * w
        if area < 100:
            continue


        cv2.rectangle(warped, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # grab character region of image
        roi = thresh[y - 5:y + h + 5, x - 5:x + w + 5]
        # # perform bitwise not to flip image
        roi = cv2.bitwise_not(roi)

        cv2.drawContours(image, [location], -1, (0, 255, 0), 3)

        text = pytesseract.image_to_string(roi, config='-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 8 --oem 3')
        clean_text = re.sub('[\W_]+', '', text)
        plate_num += clean_text

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, text=plate_num, org=(approx[0][0][0], approx[1][0][1] + 40), fontFace=font, fontScale=1,
                          color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)

    print(plate_num)
    cv2.imshow("Thresh", thresh)
    cv2.imshow("Erode", erode)
    cv2.imshow("Delation", dilation)

cv2.imshow("Edged", edged_1)
cv2.imshow("Scanned", warped)
cv2.imshow("Image", image)

cv2.waitKey(0)
cv2.destroyAllWindows()