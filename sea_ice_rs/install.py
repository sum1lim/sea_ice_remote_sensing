import subprocess
import sys


def pip_install(import_name, pkg):
    try:
        exec(f"import {import_name}")
    except ImportError:
        subprocess.call(["pip", "install", pkg])
    finally:
        exec(f"import {import_name}")


def install():
    # Install pip
    try:
        import pip
    except ImportError:
        subprocess.call(
            [sys.executable, "-m", "pip", "install", "--user", "upgrade", "pip==21.1.1"]
        )
    finally:
        import pip

    # Install 3rd party packages
    pip_install("kaggle", "kaggle")
    pip_install("cv2", "opencv-python")
    pip_install("skimage", "scikit-image")
    pip_install("tqdm", "tqdm")
    pip_install("pandas", "pandas")
    pip_install("sklearn", "sklearn")
