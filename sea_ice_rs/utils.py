import cv2
import os


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
        print("Not a valid file type")
