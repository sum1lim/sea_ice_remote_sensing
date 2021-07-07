#!/bin/bash

curl https://raw.githubusercontent.com/asylve/Sea-Ice/main/Images/Region-Grid.png --output ./data/AOIs.png
extract-colour --input ./data/AOIs.png --colour R --extension png            
threshold --input ./data/AOIs_R.png --max 255 --min 104 --extension png
connect-lines --input ./data/AOIs_R_thresh.png --iterations 2 --extension png
centroids --input ./data/AOIs_R_thresh_CL.png --max-area 700