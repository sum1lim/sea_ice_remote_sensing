import cv2
import numpy as np


def connect_lines(img, iterations):
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(img, kernel, iterations=iterations)
    img = cv2.erode(dilated, kernel, iterations=iterations)

    return img
