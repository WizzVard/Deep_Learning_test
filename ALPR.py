import cv2
import argparse
import imutils
from imutils.perspective import four_point_transform
import easyocr


ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='path to the image file')
args = vars(ap.parse_args())

image = cv2.imread(args["image"])

image = imutils.resize(image, height=500)
orig = image.copy()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
bfilter = cv2.bilateralFilter(gray, 13, 17, 17)#Noise reduction
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

warped = four_point_transform(gray, location.reshape(4, 2))

# com
# get the bounding box of the contour and
# extract the license plate from the image
(x, y, w, h) = cv2.boundingRect(location)
license_plate = gray[y:y + h, x:x + w]

reader = easyocr.Reader(['en'])
detection = reader.readtext(warped)

if len(detection) == 0:
    text = "Impossible to detect text"
    cv2.putText(image, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 3)
else:
    cv2.drawContours(image, [location], -1, (0, 255, 0), 3)
    text = detection[0][-2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    res = cv2.putText(image, text=text, org=(approx[0][0][0], approx[1][0][1] + 60), fontFace=font, fontScale=1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)

cv2.imshow("Scanned", imutils.resize(warped, width = 450))
cv2.imshow("Edged", edged_1)
cv2.imshow("Image", image)

cv2.waitKey(0)