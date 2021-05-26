import cv2
import sys
import os
import numpy as np


def output_to_window(name, image, boundaries=None):
    print(f"Image: {name}")
    image = np.array(image)

    print(f"Height: {str(len(image))}")
    print(f"Width: {str(len(image[0]))}\n")

    original_height = len(image)
    original_width = len(image[0])

    scale = 5000 / original_height

    new_height = int(original_height * scale)
    new_width = int(original_width * scale)
    dsize = (new_width, new_height)
    try:
        window_output = cv2.resize(image, dsize)
    except cv2.error:
        window_output = cv2.resize(image.astype(np.uint8), dsize)

    left = 0
    right = original_width - 1
    top = 0
    bottom = original_height - 1

    output = image.copy()

    while True:
        window_name = name + "(press q to quit)"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(
            window_name,
            window_output,
        )

        keyboard_input = cv2.waitKey(0)

        if keyboard_input == ord("q"):
            return output


def decompose_filepath(filepath):
    parent_directories = filepath.split("/")[:-1]
    indir_path = "/".join(parent_directories)
    File = filepath.split("/")[-1]
    [filename, extension] = File.split(".")

    return (indir_path, filename, extension)


def mkdir_output(inFile_path, appending_tail_string, extension, outImage):
    (inDir, filename, _) = decompose_filepath(inFile_path)

    outDir = f"{inDir}_{appending_tail_string}"
    try:
        os.mkdir(outDir)
    except FileExistsError:
        None

    try:
        cv2.imwrite(os.path.join(outDir, f"{filename}.{extension}"), outImage)

    except ValueError:
        print("Not a valid file type")
