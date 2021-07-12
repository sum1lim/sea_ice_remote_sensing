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

Also be sure to write ```python``` before the path to the script if you are on windows. 

## Centroids

Sample command (UNIX):

```centroids --input ./data/AOIs_R_thresh_CL.png --max-area 700```

## Connected Lines

Sample command (UNIX):

```connect-lines --input ./data/AOIs_R_thresh.png --iterations 2 --extension png```

## Create Datasets

Sample command on UNIX based systems for multiprocessing: 

```create-datasets --images ./data/arctic-sea-ice-image-masking/Images --masks ./data/arctic-sea-ice-image-masking/Masks --dist ./data/pixel_values.csv --patch-loc ./data/AOIs_R_thresh_CL_centroids.csv --multiprocess```

Sample command for Windows without multiprocessing:

```python scripts/create-datasets --images ./data/arctic-sea-ice-image-masking/Images --masks ./data/arctic-sea-ice-image-masking/Masks --dist ./data/pixel_values.csv --patasks --dist ./data/pixel_values.csv --patch-loc ./data/AOIs_R_thresh_CL_centroids.csv```

### Distribution Statistics 

Sample command for folder (UNIX): 

```dist-stat --input data/arctic-sea-ice-image-masking/Masks```

Sample command for single file (UNIX): 

```dist-stat --input data/arctic-sea-ice-image-masking/Masks/P0-2016042417-mask.png```

## Extract Colour 

Sample command (UNIX):

```extract-colour --input ./data/AOIs.png --colour R --extension png```         

## Extracting X-Y Coordinates of Patch Locations

For those on UNIX based OSs, there is a batch script for extracting X-Y coordinates of patch locations (see ```extract_patch_locations.sh```). 

Windows users will need to individual run the commands individually. 
### SOBEL

```SOBEL --input path/to/input/directory/or/file --contrast --extension png```

### Thresholding

```thresh --input path/to/input/directory/or/file --contrast --max 255 --min 0 --extension png```

Sample command (UNIX):

```threshold --input ./data/AOIs_R.png --max 255 --min 104 --extension png```