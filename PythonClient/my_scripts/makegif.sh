#!/bin/bash

cd ~/Gitstuff/cvlabdata2/home/kicirogl/PythonClient/my_scripts/temp_main/

ls -d 201* | sort -n

arr2=($(echo ${arr[*]}| ls -d 201* | sort -n))
cd ${arr2[${#arr2[*]}-1]}
cd a*/superimposed_images
ffmpeg -framerate 5 -i 'lift_res_%01d.png' -c:v libx264 -pix_fmt yuv420p out.mp4