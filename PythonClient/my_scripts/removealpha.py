# -*- coding: utf-8 -*-
import os
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy

for i in range(0,45):
    img1 = io.imread('test_sets/test_set_2/images/img_' + str(i)+ '.png')
    img2 = io.imread('temp_main/2018-06-13-00-38/animation_1/superimposed_images/plot3d_' + str(i)+ '.png')
    margin1 = 70
    margin2 = 30

    img1 = img1[margin2: img1.shape[0]-margin2, margin1: img1.shape[1]-margin1, :]

    img1 = scipy.ndimage.interpolation.zoom(img1, [3,3,1])

    pad_amount = int((img2.shape[0]-img1.shape[0])/2)
    img1 = np.array(img1[:,:,0:3])
    img1 = np.vstack([np.zeros([pad_amount, img1.shape[1], 3], dtype=np.uint8),img1])
    img1 = np.vstack([img1, np.zeros([pad_amount, img1.shape[1], 3], dtype=np.uint8)])

    output = np.hstack([img1, img2[:,:,0:3]])
    print(i)
    io.imsave('res2/img' + str(i)+ '.png', output)
