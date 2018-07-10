#!/bin/bash
echo Retrieving results

scp -r kicirogl@iccluster138.iccluster.epfl.ch:/cvlabdata2/home/kicirogl/PythonClient/my_scripts/temp_main/2* ~/Documents/UnrealProjects/LandscapeMountains\ 4.18/PythonClient/my_scripts/temp_main 

cd ~/Documents/UnrealProjects/LandscapeMountains\ 4.18/PythonClient/my_scripts/temp_main/

ls -d 201* | sort -n

arr2=($(echo ${arr[*]}| ls -d 201* | sort -n))
cd ${arr2[${#arr2[*]}-1]}
cd a*/images
ffmpeg -framerate 5 -i 'img_*%d.png' -c:v libx264 -pix_fmt yuv420p out.mp4