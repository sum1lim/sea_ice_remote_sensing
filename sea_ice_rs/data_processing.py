import cv2
import csv
import sys
import numpy as np
import pandas as pd
import random
import sea_ice_rs.utils as utils
from tqdm import tqdm
from datetime import datetime
from skimage.feature import greycomatrix, greycoprops


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


def GLCM_band(bordered_img, border_width, band, datapoints):
    half_right_angle = np.pi / 8

    return [
        greycomatrix(
            bordered_img[
                row : row + 2 * border_width + 1,
                col : col + 2 * border_width + 1,
                band,
            ],
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
            levels=64,
        )
        for (row, col) in datapoints
    ]

def GLCM_handler(parent_dir, csv_file, img_extension, img_dir, single_file=False):
    GLCM_dataset = open(f"{parent_dir}/GLCM.csv", "w", newline="")
    GLCM_writer = csv.writer(GLCM_dataset)

    dataframe = pd.read_csv(csv_file, header=0)

    GLCM_writer.writerow(
        list(dataframe.columns)
        + [
            "entropy_8",
            "entropy_4",
            "entropy_3",
            "ASM_8",
            "ASM_4",
            "ASM_3",
            "contrast_8",
            "contrast_4",
            "contrast_3",
            "homogeneity_8",
            "homogeneity_4",
            "homogeneity_3",
            "dissimilarity_8",
            "dissimilarity_4",
            "dissimilarity_3",
        ]
    )

    grouped = dataframe.groupby(["patch_num", "year", "DOY", "hour"])

    if (single_file == True):
        for name, group in grouped:

            data_points = [
                (int(item["pix_loc_y"]), int(item["pix_loc_x"]))
                for idx, item in group.iterrows()
            ]

            GLCM_matrices = generate_GLCM(f"{img_dir}", data_points)

            entropy = glcm_product(GLCM_matrices, "entropy")
            ASM = glcm_product(GLCM_matrices, "ASM")
            contrast = glcm_product(GLCM_matrices, "contrast")
            homogeneity = glcm_product(GLCM_matrices, "homogeneity")
            dissimilarity = glcm_product(GLCM_matrices, "dissimilarity")

            i = 0
            for idx, item in group.iterrows():
                GLCM_features = (
                    [item[i] for i in range(len(item))]
                    + entropy[i, :].tolist()
                    + ASM[i, :].tolist()
                    + contrast[i, :].tolist()
                    + homogeneity[i, :].tolist()
                    + dissimilarity[i, :].tolist()
                )
                GLCM_writer.writerow(GLCM_features)
                i += 1
        return

    for name, group in tqdm(grouped):
        patch_num = int(name[0])
        year = int(name[1])

        month = datetime.strptime(f"{year} {name[2]}", "%Y %j").strftime("%m")
        day = datetime.strptime(f"{year} {name[2]}", "%Y %j").strftime("%d")
        hour = "{:0>2}".format(int(name[3]))

        img_file = f"P{patch_num}-{year}{month}{day}{hour}.{img_extension}"

        data_points = [
            (int(item["pix_loc_y"]), int(item["pix_loc_x"]))
            for idx, item in group.iterrows()
        ]

        GLCM_matrices = generate_GLCM(f"{img_dir}/{img_file}", data_points)

        entropy = glcm_product(GLCM_matrices, "entropy")
        ASM = glcm_product(GLCM_matrices, "ASM")
        contrast = glcm_product(GLCM_matrices, "contrast")
        homogeneity = glcm_product(GLCM_matrices, "homogeneity")
        dissimilarity = glcm_product(GLCM_matrices, "dissimilarity")

        i = 0
        for idx, item in group.iterrows():
            GLCM_features = (
                [item[i] for i in range(len(item))]
                + entropy[i, :].tolist()
                + ASM[i, :].tolist()
                + contrast[i, :].tolist()
                + homogeneity[i, :].tolist()
                + dissimilarity[i, :].tolist()
            )
            GLCM_writer.writerow(GLCM_features)
            i += 1

def generate_GLCM(inFile, datapoints):
    inImage = cv2.imread(inFile)

    rescaled_img = ((inImage / 255) * (64 - 1)).astype(int)

    border_width = 5
    bordered_img = cv2.copyMakeBorder(
        rescaled_img,
        border_width,
        border_width,
        border_width,
        border_width,
        borderType=cv2.BORDER_REFLECT_101,
    )

    GLCM_0 = GLCM_band(bordered_img, border_width, 0, datapoints)
    GLCM_1 = GLCM_band(bordered_img, border_width, 1, datapoints)
    GLCM_2 = GLCM_band(bordered_img, border_width, 2, datapoints)

    return [GLCM_0, GLCM_1, GLCM_2]


def generate_entropy(GLCM):
    e = np.finfo(float).eps

    return [
        np.sum(-np.multiply(GLCM[:, :, :, i], np.log(GLCM[:, :, :, i] + e)))
        for i in range(GLCM.shape[-1])
    ]


def glcm_product(GLCM_matrices, product_type):
    return np.transpose(
        np.asarray(
            [
                [
                    np.sum(generate_entropy(GLCM))
                    if product_type == "entropy"
                    else np.sum(greycoprops(GLCM, product_type)[0])
                    for GLCM in GLCM_matrices[i]
                ]
                for i in range(len(GLCM_matrices))
            ]
        )
    )


def normalize(input, std_data):
    tr_df = pd.read_csv(std_data)

    minimums = {col: tr_df[col].min() for col in tr_df.columns if col != "label"}
    maximums = {col: tr_df[col].max() for col in tr_df.columns if col != "label"}

    df = pd.read_csv(input)

    for col in df.columns:
        if col == "label":
            continue
        df[col] = (df[col] - minimums[col]) / (maximums[col] - minimums[col])

    return df

def get_count_of_pixel_classes(d, img):
    """
    Update the count of pixel value classes based on image reading
    """
    unique, counts = np.unique(img, return_counts=True)

    for key, value in dict(zip(unique, counts)).items():
        if key in d:
            d[key] += value
        else:
            d[key] = value


def sampling_probability(mask_dir, images):
    """
    The function sets the probabilitiy of a sample to be included in the dataset.
    A sample in the class with a higher occurrence is less likely to be selected.
    The class with the smallest occurrence will have all of its samples included.
    The expected number of samples per class is equal throughout the entire data.
    """
    dict_of_ppv = {}
    for img_f in images:
        _, img_name, _ = utils.decompose_filepath(img_f)
        file_path = f"{mask_dir}/{img_name}-mask.png"
        try:  # Valid image file
            inImage = cv2.imread(file_path, 0)
            get_count_of_pixel_classes(dict_of_ppv, inImage)
        except:  # non-image file
            print(f"Error occurred when processing {img_f}")

    try:
        del dict_of_ppv[None]
    except KeyError:
        None

    min_count = min(dict_of_ppv.values())
    return {label: (min_count / count) / 5 for label, count in dict_of_ppv.items()}


def patch_location_map(patch_loc_file):
    """
    Read (X, Y) coordinates of the patches.
    """
    with open(patch_loc_file, "r") as patch_locs:
        patch_loc_reader = csv.reader(patch_locs)
        return {row[0][1:]: (row[1], row[2]) for row in patch_loc_reader}


def sampling(
    images,
    dataset_file,
    img_dir,
    mask_dir,
    prob_dict,
    patch_loc_dict,
    pbar_text,
):
    """
    Sample the data using the probabilities defined.
    """

    # Write headers
    headers = [
        "label",
        "patch_num",
        "year",
        "patch_loc_y",
        "patch_loc_x",
        "DOY",
        "hour",
        "pix_loc_y",
        "pix_loc_x",
        "band_8",
        "band_4",
        "band_3",
    ]
    dataset = open(dataset_file, "w", newline="")
    csv_writer = csv.writer(dataset)
    csv_writer.writerow(headers)

    data_summary = {}

    # Sample from images
    pbar = tqdm(images)
    for img in pbar:
        pbar.set_description(f"{pbar_text}: {img}")
        _, filename, extension = utils.decompose_filepath(img)
        if extension != "jpg":
            continue

        patch_num = filename.split("-")[0][1:]

        # Extract date information
        date_info = filename.split("-")[1]
        year = int(date_info[0:4])
        month = int(date_info[4:6])
        day = int(date_info[6:8])
        hour = int(date_info[8:10])

        doy = int(datetime(year, month, day).strftime("%j"))

        inImage = cv2.imread(f"{img_dir}/{img}")
        inMask = cv2.imread(f"{mask_dir}/{filename}-mask.png")

        for row in range(inImage.shape[0]):
            for col in range(inImage.shape[1]):
                label = inMask[row][col][0]
                sampling_weights = [1 - prob_dict[label], prob_dict[label]]
                selection = random.choices(["skip", "sample"], sampling_weights, k=1)[0]

                if selection == "skip":
                    continue
                try:
                    data_summary[label] += 1
                except KeyError:
                    data_summary[label] = 1

                pix_vals = inImage[row][col]
                sample = [
                    label,
                    patch_num,
                    year,
                    patch_loc_dict[patch_num][0],
                    patch_loc_dict[patch_num][1],
                    doy,
                    hour,
                    row,
                    col,
                    pix_vals[0],
                    pix_vals[1],
                    pix_vals[2],
                ]

                csv_writer.writerow(sample)

    print(f"{pbar_text} thread finished", sys.stdout)
    dataset.close()

    for label, count in data_summary.items():
        print(f"{label}: {count}")

    print(f"SUM: {sum(data_summary.values())}")
