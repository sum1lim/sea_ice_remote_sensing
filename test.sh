#!/bin/bash

for result_dir in ./results/*; do 
    if [ -d "$result_dir" ]; then
        echo "************************* Testing on $result_dir *************************" 
        tmp=${result_dir:10:100}
        config_name=${tmp#*_}
        test-model --dl-config ./DL_configs/${config_name}.yml --result-dir $result_dir
    fi
done