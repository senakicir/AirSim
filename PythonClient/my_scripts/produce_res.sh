#!/bin/bash
cd res2
ffmpeg -framerate 5 -i 'img%01d.png' -c:v libx264 -pix_fmt yuv420p anim1_overall.mp4

cd res2/calib
ffmpeg -framerate 5 -i 'img%01d.png' -c:v libx264 -pix_fmt yuv420p anim1_calib_v2.mp4
