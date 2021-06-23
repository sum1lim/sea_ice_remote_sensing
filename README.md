# Sea Ice Remote Sensing

**Authors:** [Sangwon Lim](https://github.com/sum1lim) and [Omar Kawach](https://github.com/omarkawach)

**Purpose:** Submission for the group project in University of Victoria's Artificial Intelligence course (ECE 470). 

## Development Instructions 

### Getting the Data

For our research we used optical data from Kaggle. Our Python programs in the package are intended to run the following dataset:

[Sylvester, S. (2021, April). Arctic sea ice image masking, Version 3. Retrieved May 17, 2021](https://www.kaggle.com/alexandersylvester/arctic-sea-ice-image-masking/version/3)

To use the dataset we selected, ensure that you have a [Kaggle API token](https://www.kaggle.com/c/two-sigma-financial-news/discussion/83593) properly saved locally. 

Then you may ```cd``` into the ```data``` folder and run in venv

```
kaggle datasets download alexandersylvester/arctic-sea-ice-image-masking 
```

Once you have the zipped file, unzip it. 

**Note**: You will need to have Kaggle already pip installed. 

### Install the Package in a Python Virtual Environment

#### MacOS / Linux

```
python -m venv venv
source venv/bin/activate
pip install .
```

#### Windows

```
python -m venv venv
venv/Scripts/activate
pip install .
```

## Running Programs

Make sure you are in the main Python working directory when running the commands in windows.

### SOBEL
```SOBEL --input path/to/input/directory/or/file --contrast --extension png```

### Thresholding
```thresh --input path/to/input/directory/or/file --contrast --max 255 --min 0 --extension png```

### Distribution Statistics 

Sample command for folder: 

```dist-stat --input data/arctic-sea-ice-image-masking/Masks```

Sample command for single file: 

```dist-stat --input data/arctic-sea-ice-image-masking/Masks/P0-2016042417-mask.png```