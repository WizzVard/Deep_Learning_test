from imutils import contours
import numpy as np
import argparse
import imutils
import cv2
from skimage.segmentation import clear_border


ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required=True, help="path to the video file")
ap.add_argument("-c", "--clip", type=float, default=2.0, help="threshold for contrast limiting")
ap.add_argument("-t", "--tile", type=int, default=8, help="tile grid size -- divides image into tile x time cells")
args = vars(ap.parse_args())

vid = cv2.VideoCapture(args["video"])

while(True):
    # grab the current frame and initialize the status text
    (grabbed, frame) = vid.read()
    status = "No Targets"

    # check to see if we have reached the end of the video
    if not grabbed:
        break

    (h, w, d) = frame.shape
    r = 500.0 / w
    dim = (500, int(h * r))
    image = cv2.resize(frame, dim)

    # define the list of boundaries
    boundaries = [
        ([25, 0, 180], [254, 254, 255]),
    ]


    # loop over the boundaries
    for (lower, upper) in boundaries:
      # create NumPy arrays from the boundaries
      lower = np.array(lower, dtype = "uint8")
      upper = np.array(upper, dtype = "uint8")
      # find the colors within the specified boundaries and apply
      # the mask
      mask = cv2.inRange(image, lower, upper)
      output = cv2.bitwise_and(image, image, mask = mask)
      # show the images
      cv2.imshow("images", np.hstack([image, output]))

    gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)[1]
    # perform a series of erosions and dilations
    # thresh = cv2.erode(thresh, None, iterations = 4)
    # thresh = cv2.dilate(thresh, None, iterations = 4)


    # find contours in the thresholded image, then initialize the
    # digit contours lists
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    fish = []
    # loop over the digit area candidates
    count = 0
    for c in cnts:
        # compute the bounding box of the contour
        (x, y, w, h) = cv2.boundingRect(c)
        # if the contour is sufficiently large, it must be a digit
        count += 1
        if count <= round(len(cnts)/2):
            if (w >= 0 and w <= 0) and (h >= 20 and h <= 40):
                fish.append(c)
        else:
            if (w >= 10 and w <= 60) and (h >= 20 and h <= 80):
                fish.append(c)
    for c in fish:
        # extract the digit ROI
        (x, y, w, h) = cv2.boundingRect(c)
    cv2.rectangle(image, (x, y), (x+w, y+h), (255,0,0), 2)
    cv2.imshow("im", image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()


