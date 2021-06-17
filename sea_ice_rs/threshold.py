import sys
import cv2


def threshold(img, max_val=None, min_val=None):
    while not ((max_val or min_val) and min_val < max_val):
        if not (max_val or min_val):
            print(
                "At least one of maximum or minimum threshold values has to be provided"
            )
        else:
            print("Minimum threshold value should not exceed maximum threshold value")
        try:
            max_val = int(input("Max: "))
            min_val = int(input("Min: "))
        except ValueError:
            print("Threshold values should be integers")
            max_val = None
            min_val = None

    if max_val:
        img[img > max_val] = 0

    if min_val:
        img[img < min_val] = 0

    img[img > 0] = 255

    return img
