# Sea Ice Remote Sensing

**Authors**: [Sangwon Lim](https://github.com/sum1lim) and [Omar Kawach](https://github.com/omarkawach)

**Purpose**: Submission for the group project in University of Victoria's Artificial Intelligence course (ECE 470). 

**Description**: Get a model and see if it can be applicable to other data.

**Note**: Couldn't seem to get TensorFlow running on Windows or Ubuntu for the ML related scripts. For now MacOS with a M1 chip seems to work.

## Getting Started

### Install the Package in a Python Virtual Environment

To avoid conflicts, the first step is to isolate this project by creating a Python virtual environment called ```venv```. The virtual environment will have it's own python interpreter, dependencies, and scripts. Commands should only be entered in a terminal that has ```venv``` active. 

#### MacOS / Linux

```
python -m venv venv
source venv/bin/activate
pip install .
pip install -r requirements.txt
```

#### Windows

```
python -m venv venv
venv/Scripts/activate
pip install .
pip install -r requirements.txt
```

### Getting the Data

For our research we used optical data from Kaggle. The Python programs in the package were built with the following dataset:

[Sylvester, S. (2021, April). Arctic sea ice image masking, Version 3. Retrieved May 17, 2021](https://www.kaggle.com/alexandersylvester/arctic-sea-ice-image-masking/version/3)

To use the dataset we selected, ensure that you have a [Kaggle API token](https://www.kaggle.com/c/two-sigma-financial-news/discussion/83593) properly saved locally. 

Once you have ensured that you have a Kaggle API token, ```cd``` into the ```data``` folder and run the following command:

```
kaggle datasets download alexandersylvester/arctic-sea-ice-image-masking 
```

## Running Programs

For the workflow, the commands below are in sequential order. Again, make sure you are in ```venv``` when running these commands.

**Note**: If you are on Windows, be sure to write ```python``` before the path to the script you are trying to run. Only Windows requires a relative path to the script you are trying to run. The commands below assume you are in the project's home directory. 

### 1. Extracting X-Y Coordinates of Patch Locations

***Purpose***: Preprocessing step for feature extraction.

**Note**: The reference image of patch locations is retrieved using the ```extract_patch_locations``` shell script or batch script. 

##### Unix-like Operating Systems

```
./extract_patch_locations.sh
```

##### Windows

```
./extract_patch_locations.bat
```

### 2. Pixel Based Feature Extraction

***Purpose***: Features for machine learning should be extracted for each sample pixel. Extracts pixel samples where the number of samples per class is nearly consistent throughout the resulting dataset.

#### Distribution statistics 

##### Unix-like Operating Systems

Command to run distribution statistics on a folder: 

```
dist-stat --input data/arctic-sea-ice-image-masking/Masks
```

Command to run distribution statistics on a single file: 

```
dist-stat --input data/arctic-sea-ice-image-masking/Masks/P0-2016042417-mask.png
```

##### Windows

Command to run distribution statistics on a folder: 

```
python scripts/dist-stat --input data/arctic-sea-ice-image-masking/Masks
```

Command to run distribution statistics on a single file: 

```
python scripts/dist-stat --input data/arctic-sea-ice-image-masking/Masks/P0-2016042417-mask.png
```

#### Create datasets

##### Unix-like Operating Systems

Command to create datasets via multiprocessing:

```
create-datasets --images ./data/arctic-sea-ice-image-masking/Images --masks ./data/arctic-sea-ice-image-masking/Masks --dist ./data/pixel_values.csv --patch-loc ./data/AOIs_R_thresh_CL_centroids.csv --multiprocess
```

Command to create datasets without multiprocessing:

```
create-datasets --images ./data/arctic-sea-ice-image-masking/Images --masks ./data/arctic-sea-ice-image-masking/Masks --dist ./data/pixel_values.csv --dist ./data/pixel_values.csv --patch-loc ./data/AOIs_R_thresh_CL_centroids.csv
```

##### Windows

**Note:** WinError 5 will occur if you try creating datasets with multiprocessing on Windows.

Command to create datasets without multiprocessing:

```
python scripts/create-datasets --images ./data/arctic-sea-ice-image-masking/Images --masks ./data/arctic-sea-ice-image-masking/Masks --dist ./data/pixel_values.csv --dist ./data/pixel_values.csv --patch-loc ./data/AOIs_R_thresh_CL_centroids.csv
```

### 3. Generate GLCM Texture Features

***Purpose***: Generate 5 GLCM products for each of the data points

##### Unix-like Operating Systems

```
GLCM --input ./data/train_dataset/raw.csv --img-dir ./data/arctic-sea-ice-image-masking/Images
```

##### Windows

```
python scripts/GLCM --input ./data/train_dataset/raw.csv --img-dir ./data/arctic-sea-ice-image-masking/Images
```

### 4. Normalize Data

***Purpose***: Normalizing data can result in better performance of the model. Except for training data, the strategy of normalization should include standard min & max values instead of calculating such values within the dataset. The standard values are from the training dataset.

##### Unix-like Operating Systems

To normalize the training dataset:

```
normalize --input ./data/train_dataset/GLCM.csv --std-data ./data/train_dataset/GLCM.csv
```

To normalize the test dataset:

```
normalize --input ./data/test_dataset/GLCM.csv --std-data ./data/train_dataset/GLCM.csv
```

##### Windows

To normalize the training dataset:

```
python scripts/normalize --input ./data/train_dataset/GLCM.csv --std-data ./data/train_dataset/GLCM.csv
```

To normalize the test dataset:

```
python scripts/normalize --input ./data/test_dataset/GLCM.csv --std-data ./data/train_dataset/GLCM.csv
```

### 5. Machine Learning

***Purpose***: Training, testing, and predicting of the model.

**Note**: The commands below only seem to work on MacOS with M1 chip.

#### Neural Network

Train neural network:

```
neural-network --dl-config ./DL_configs/GLCM_C6_cat.yml
```
![Screenshot from 2021-09-15 12-38-28](https://user-images.githubusercontent.com/58998142/133499033-a8baf7c3-fa64-4152-8b65-80451c361ad5.png)

#### CNN

Train 1D-CNN (To concatenate multi-layer neural network, add features other than spectral data and GLCM products):

```
# 1D-CNN
CNN --dl-config ./DL_configs/GLCM_C6.yml
```
![Screenshot from 2021-09-15 12-38-29](https://user-images.githubusercontent.com/58998142/133499115-1151c202-cf3a-48fc-a1ef-d6a67e44e7e7.png)
![Screenshot from 2021-09-15 12-38-30](https://user-images.githubusercontent.com/58998142/133499126-f3228c1f-9aea-48c7-ba24-61b7ca3f2814.png)

```
# Concatenation of 1D-CNN and multi-layer NN
CNN --dl-config ./DL_configs/GLCM_C6_cat.yml
```
![Screenshot from 2021-09-15 12-38-31](https://user-images.githubusercontent.com/58998142/133499138-a5faa969-1cd7-4022-831e-a087dc168601.png)

#### Test Model

Test the model:

```
test-model --dl-config ./DL_configs/GLCM_C6_cat.yml --result-dir ./results/CNN_GLCM_C6_cat
```

#### Predict

For an image, run a prediction:

```
predict --patch-loc ./data/AOIs_R_thresh_CL_centroids.csv --std-data ./data/train_dataset/GLCM.csv --result-dir ./results/CNN_GLCM_C4_cat/ckpt_1 --dl-config ./DL_configs/GLCM_C4_cat.yml --mask-dir ./data/arctic-sea-ice-image-masking/Masks --input ./data/arctic-sea-ice-image-masking/Images/P54-2018071616.jpg --classes 4
```
![Figure_4](https://user-images.githubusercontent.com/58998142/128421169-49298419-2d63-4cdd-a121-a5bb25f9d212.png)

Figure 1. Expert Image

![Figure_2](https://user-images.githubusercontent.com/58998142/128421197-01e1bcc8-e85d-4259-850b-f66d2b7406d5.png)

Figure 2. Prediction Image


## Sources 

[1] R. Ressel, A. Frost and S. Lehner, "A Neural Network-Based Classification for Sea Ice Types on X-Band SAR Images," in IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, vol. 8, no. 7, pp. 3672-3680, July 2015, doi: 10.1109/JSTARS.2015.2436993.
