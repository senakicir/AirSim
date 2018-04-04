# -*- coding: utf-8 -*-
import os
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy

for i in range(125,151):
    original_image = io.imread('temp_main/2018-02-14-18-00/images/img_' + str(i)+ '.png')
    image = np.array(original_image[:,:,0:3])
    io.imsave('temp_main/2018-02-14-18-00/images_new/img' + str(i)+ '.png', image)