import cv2
import numpy as np
from tqdm import tqdm
from skimage.feature import greycomatrix


def contrast(inImage):
    """
    The functinon spreads the concentrated pixel values to 0~255 range
    """
    minValue = np.min(inImage)
    maxValue = np.max(inImage)

    data_range = maxValue - minValue

    outImage = np.int_((255 * (inImage - minValue)) / data_range)

    return outImage.astype(np.uint8)


def threshold(img, max_val=None, min_val=None):
    ret_img = img.copy()
    while not ((max_val or min_val) and min_val <= max_val):
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
        ret_img[ret_img > max_val] = 0

    if min_val:
        ret_img[ret_img < min_val] = 0

    ret_img[ret_img > 0] = 255

    return ret_img


def connect_lines(img, iterations):
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(img, kernel, iterations=iterations)
    img = cv2.erode(dilated, kernel, iterations=iterations)

    return img


def extract_colour(img, colour):

    pix_val_sum = img.sum(axis=2)

    if colour == "R":
        output_img = np.divide(img[:, :, 2], pix_val_sum)

    elif colour == "G":
        output_img = np.divide(img[:, :, 1], pix_val_sum)

    elif colour == "B":
        output_img = np.divide(img[:, :, 0], pix_val_sum)

    else:
        raise ValueError("Invalid colour. Select from ['R', 'G', 'B']")

    output_img = np.nan_to_num(output_img)

    return np.dstack([output_img])


def generate_GLCM(inFile):
    inImage = cv2.imread(inFile)

    rescaled_img = ((inImage / 255) * (32 - 1)).astype(int)

    img_border = cv2.copyMakeBorder(
        rescaled_img[:, :, 0], 5, 5, 5, 5, borderType=cv2.BORDER_REFLECT_101
    )

    num_rows = inImage.shape[0]
    num_cols = inImage.shape[1]

    GLCM_matrices = []

    for row in tqdm(range(5, num_rows)):
        GLCM_row = []
        for col in range(5, num_cols):
            patch = img_border[row - 5 : row + 6, col - 5 : col + 6]
            half_right_angle = np.pi / 8
            histogram = greycomatrix(
                patch,
                distances=[1],
                angles=[
                    0,
                    half_right_angle,
                    2 * half_right_angle,
                    3 * half_right_angle,
                    4 * half_right_angle,
                    5 * half_right_angle,
                    6 * half_right_angle,
                    7 * half_right_angle,
                ],
                levels=32,
            )

            GLCM_row.append(histogram)

        GLCM_matrices.append(GLCM_row)

    return GLCM_matrices
