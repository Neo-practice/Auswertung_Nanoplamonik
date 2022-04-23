import numpy as np
import matplotlib.pyplot as plt

# import the necessary packages
from imutils import contours
from skimage import measure
import argparse
import imutils
import cv2

def scale(bild):
    height, width = bild.shape # (height, width)
    factor = np.minimum(1080 / height, 1920 / width)
    return cv2.resize(bild, (int(width*factor), int(height *factor)))


# construct the argument parse and parse the arguments


# load the image, convert it to grayscale, and blur it
image = cv2.imread("test_silber_df_without_pol.png")




gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (11, 11), 0)

# threshold the image to reveal light regions in the
# blurred image
# thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)[1]
img = gray

# cv2.imshow("thresh", img)

roi = cv2.selectROI(img)
roi_cropped = img[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]
# roi2 = cv2.selectROI(roi_cropped)
roi2 = cv2.selectROI(scale(roi_cropped))


# cv2.imshow("test", scale(roi2, factor))
# cv2.imshow("notwhite", roi)
# cv2.imshow("Zoom in",roi_cropped)
# cv2.imshow("Roi", roi2)



cv2.waitKey(0)