import cv2
import os
import sys
import numpy as np
from tqdm import tqdm
from .contrast import contrast
from .threshold import threshold


def decompose_filepath(filepath):
    """
    decompose filepath into three components:
    directory path, file name and extension
    """
    parent_directories = filepath.split("/")[:-1]
    if parent_directories[-1] == "":
        del parent_directories[-1]
    dir_path = "/".join(parent_directories)
    File = filepath.split("/")[-1]
    [filename, extension] = File.split(".")

    return (dir_path, filename, extension)


def output(output_name, image, split_rgb=False):
    """
    stores the image to the given path.
    split_rgb option allows to store each band separately
    """
    if image.shape[2] == 1 or not split_rgb:
        cv2.imwrite(output_name, image)
    else:
        for i in range(image.shape[2]):
            (dir_path, filename, extension) = decompose_filepath(output_name)
            cv2.imwrite(
                os.path.join(dir_path, f"{filename}({i}).{extension}"), image[:, :, i]
            )


def output_to_window(name, image, orginal_img=None):
    """
    output image to an interactive window
    """
    print(f"Image: {name}")
    if orginal_img is None:
        orginal_img = image.copy()

    while True:
        window_name = (
            f"{name} (press q to quit; r to revert; c to contrast; t to threshold)"
        )
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(
            window_name,
            image,
        )

        keyboard_input = cv2.waitKey(0)

        if keyboard_input == ord("r"):
            image = orginal_img

        if keyboard_input == ord("c"):
            image = contrast(image)

        if keyboard_input == ord("t"):
            image = orginal_img.copy()
            image = threshold(image)

        if keyboard_input == ord("q"):
            return image


def mkdir_output(
    inFile_path, appending_tail_string, extension, outImage, split_rgb=False
):
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
        output(
            os.path.join(outDir, f"{filename}.{extension}"),
            outImage,
            split_rgb=split_rgb,
        )

    except ValueError:
        print("Not a valid file type", file=sys.stderr)


def process_multiple_inputs(
    input_dir, tail_str, extension, func, params=None, contrast_bool=False
):
    """
    utility function to handle a directory input
    """
    for img_f in tqdm(os.listdir(input_dir)):
        file_path = f"{input_dir}/{img_f}"
        try:  # Valid image file
            inImage = cv2.imread(file_path)

            if params:
                processed = func(inImage, *params)
            else:
                processed = func(inImage)

            if contrast_bool:  # enhances contrast of the output image
                processed = contrast(processed).astype(np.uint8)

            if extension:  # save to an output file if extension is given
                mkdir_output(
                    file_path, tail_str, extension, processed, split_rgb=True
                )  # split_rgb option saves an image per band

        except KeyboardInterrupt:
            sys.exit(1)

        except:  # non-image file
            print(f"Error occurred when processing {img_f}")


def process_single_input(
    input_file,
    tail_str,
    extension,
    func,
    params=None,
    contrast_bool=False,
    split_rgb=False,
):
    """
    utility function to handle a single file input
    """
    try:
        img = cv2.imread(input_file)
    except:
        print("The file does not exist", file=sys.stderr)
        exit(1)

    if params:
        processed = func(img, *params)
    else:
        processed = func(img)

    if contrast_bool:  # enhances contrast of the output image
        processed = contrast(processed).astype(np.uint8)

    (inDir, filename, _) = decompose_filepath(input_file)
    output_name = f"{filename}_{tail_str}"
    for i in range(
        processed.shape[2]
    ):  # for loop deals with multiple bands in the image
        output_to_window(f"{output_name}(band {i})", processed[:, :, i])

    if extension:  # save to an output file if extension is given
        outputFile = f"{inDir}/{output_name}.{extension}"
        try:
            output(outputFile, processed, split_rgb)
        except ValueError:
            print("Not a valid file type")
