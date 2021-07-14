import cv2
import csv
import numpy as np
import random
import sea_ice_rs.utils as utils
from tqdm import tqdm


def contrast(inImage):
    """
    The functinon spreads the concentrated pixel values to 0~255 range
    """
    minValue = np.min(inImage)
    maxValue = np.max(inImage)

    data_range = maxValue - minValue

    outImage = np.int_((255 * (inImage - minValue)) / data_range)

    return outImage


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


def sampling_probability(dist_stats_file):
    """
    The function sets the probabilitiy of a sample to be included in the dataset.
    A sample in the class with a higher occurrence is less likely to be selected.
    The class with the smallest occurrence will have all of its samples included.
    The expected number of samples per class is equal throughout the entire data.
    """
    with open(dist_stats_file, "r") as dist_file:
        dist_file_reader = csv.reader(dist_file)
        dist_stats = [row for row in dist_file_reader]
        labels = dist_stats[0]
        counts = list(map(float, dist_stats[1]))

        null_idx = labels.index("")
        del labels[null_idx]
        del counts[null_idx]

        return {int(labels[i]): min(counts) / counts[i] for i in range(len(labels))}


def patch_location_map(patch_loc_file):
    """
    Read (X, Y) coordinates of the patches.
    """
    with open(patch_loc_file, "r") as patch_locs:
        patch_loc_reader = csv.reader(patch_locs)
        return {row[0][1:]: (row[1], row[2]) for row in patch_loc_reader}


def sampling(
    images,
    tr_writer,
    te_writer,
    img_dir,
    mask_dir,
    prob_dict,
    patch_loc_dict,
    thread_num,
):
    """
    Sample the data using the probabilities defined.
    """
    random.shuffle(images)
    pbar = tqdm(images)
    for idx, img in enumerate(pbar):
        if idx < 0.8 * len(pbar):
            dataset = "Train"
        else:
            dataset = "Test"

        pbar.set_description(f"{dataset}: {img}")
        _, filename, extension = utils.decompose_filepath(img)
        if extension != "jpg":
            return
        patch_num = filename.split("-")[0][1:]
        year = filename.split("-")[1][0:4]
        month = filename.split("-")[1][4:6]
        day = filename.split("-")[1][6:8]
        hour = filename.split("-")[1][8:10]

        inImage = cv2.imread(f"{img_dir}/{img}")
        inMask = cv2.imread(f"{mask_dir}/{filename}-mask.png")

        for row in range(inImage.shape[0]):
            for col in range(inImage.shape[1]):
                label = inMask[row][col][0]
                sampling_weights = [1 - prob_dict[label], prob_dict[label] * 0.8]
                selection = random.choices(["skip", "sample"], sampling_weights, k=1)[0]

                if selection == "skip":
                    continue
                elif selection == "sample":
                    pix_vals = inImage[row][col]
                    sample = [
                        label,
                        patch_num,
                        patch_loc_dict[patch_num][0],
                        patch_loc_dict[patch_num][1],
                        year,
                        month,
                        day,
                        hour,
                        row,
                        col,
                        pix_vals[0],
                        pix_vals[1],
                        pix_vals[2],
                    ]

                if dataset == "Train":
                    tr_writer.writerow(sample)
                else:
                    te_writer.writerow(sample)

    print(f"Thread #{thread_num} done", file=sys.stdout)
