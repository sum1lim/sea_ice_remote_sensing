import numpy as np
import sys
import cv2
from skimage import filters


def generate_SOBEL(inFile):
    try:
        inImage = cv2.imread(inFile)
    except:
        print("No such file or directory", file=sys.stderr)
        exit(1)

    filtered = filters.sobel(inImage)

    return filtered
