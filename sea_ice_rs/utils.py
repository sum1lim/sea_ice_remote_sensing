import cv2
import sys
import os
import numpy as np


def output_to_window(name, image):
    """
    output image to an interactive window
    """
    print(f"Image: {name}")

    while True:
        window_name = name + "(press q to quit)"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(
            window_name,
            image,
        )

        keyboard_input = cv2.waitKey(0)

        if keyboard_input == ord("q"):
            return


def decompose_filepath(filepath):
    """
    decompose filepath into three components:
    directory path, file name and extension
    """
    parent_directories = filepath.split("/")[:-1]
    dir_path = "/".join(parent_directories)
    File = filepath.split("/")[-1]
    [filename, extension] = File.split(".")

    return (dir_path, filename, extension)


def mkdir_output(inFile_path, appending_tail_string, extension, outImage):
    """
    make a new diretory and store the image in it
    """
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
