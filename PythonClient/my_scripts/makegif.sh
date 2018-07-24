#!/bin/bash

cd ~/Gitstuff/cvlabdata2/home/kicirogl/PythonClient/my_scripts/temp_main/

ls -d 201* | sort -n

arr2=($(echo ${arr[*]}| ls -d 201* | sort -n))
cd ${arr2[${#arr2[*]}-1]}
cd t*/superimposed_images
ffmpeg -framerate 5 -i 'lift_res_%01d.png' -c:v libx264 -pix_fmt yuv420p lift_res.mp4
ffmpeg -framerate 5 -i 'openpose_%01d.png' -c:v libx264 -pix_fmt yuv420p -vf pad="width=iw:height=ih+1:x=0:y=0:color=white" openpose.mp4