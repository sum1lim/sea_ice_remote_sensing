@echo off

curl https://raw.githubusercontent.com/asylve/Sea-Ice/main/Images/Region-Grid.png -o ./data/AOIs.png
python scripts/extract-colour --input ./data/AOIs.png --colour R --extension png     
python scripts/threshold --input ./data/AOIs_R.png --max 255 --min 104 --extension png
python scripts/connect-lines --input ./data/AOIs_R_thresh.png --iterations 2 --extension png
python scripts/centroids --input ./data/AOIs_R_thresh_CL.png --max-area 700