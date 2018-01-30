# -*- coding: utf-8 -*-
import torch
import torchvision
import os
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from skimage import io, transform
import torchvision.models as models
import pandas as pd
import numpy as np
from resnet_VNECT_heat import resnet50 
import IPython
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy

bones_h36m = [[0, 1], [1, 2], [2, 3],
             [0, 4], [4, 5], [5, 6],
             [0, 7], [7, 8], [8, 9], [9, 10],
             [8, 14], [14, 15], [15, 16],
             [8, 11], [11, 12], [12, 13],]
INPUT_IMAGE_SIZE = [720, 1280]
NUM_OF_JOINTS = 17

PRETRAINED = True
config_dict = {'img_mean' : (0.485, 0.456, 0.406), 'img_std' : (0.229, 0.224, 0.225)}

def initNet():
    net = resnet50(PRETRAINED).cuda()
    return net

def FindJointPos2D(photo, num_of_photos, net, make_plot = True):

    class DroneFootage(Dataset):
        def __init__(self, input_data, transform=None):
            self.transform = transform
            self.input_data = input_data
        
        def __len__(self):
            return 1

        def __getitem__(self, idx):
            image = self.input_data
            image = np.array(image[:,:,0:3], dtype='f')
            images = [scipy.misc.imresize( image, 1.0), scipy.misc.imresize( image, 0.8), scipy.misc.imresize( image, 1.2)]
            samples = []
            for an_image in images:
                if self.transform:
                    image = self.transform(an_image)
                samples.append({'image': image})
            return samples

    output_dir = '/cvlabdata2/home/kicirogl/activehumanposeest/output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    drone_test_dataset = DroneFootage(input_data=photo, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(config_dict['img_mean'], config_dict['img_std'])]))
    testloader = torch.utils.data.DataLoader(drone_test_dataset, batch_size=1, shuffle=False, num_workers=1)

    fig = plt.figure()
    for i, samples in enumerate(testloader, 0):
        resized_outputs = []
        for image_num, data in enumerate(samples):
            images = data['image']
            
            if image_num == 0:
                an_image = scipy.misc.imresize(np.squeeze(images.numpy()).transpose([1,2,0]), INPUT_IMAGE_SIZE)
            img_dict = {'img_crop': Variable(images.cuda())}
            outputs = net(img_dict)
            output_heatmaps = outputs['2d_heat']

        
            for output in output_heatmaps:
                temp = output.data.cpu().numpy()
                an_output = np.zeros([INPUT_IMAGE_SIZE[0],INPUT_IMAGE_SIZE[1],NUM_OF_JOINTS])
                for heatmap in range(0,NUM_OF_JOINTS):
                    an_output[:,:,heatmap] = scipy.misc.imresize(temp[heatmap,:,:], INPUT_IMAGE_SIZE)
                resized_outputs.append(an_output)

        mean_heatmap = (resized_outputs[0] + resized_outputs[1] + resized_outputs[2])/3

        bones = np.zeros((2,NUM_OF_JOINTS))
        for heatmap in range(0,NUM_OF_JOINTS):
            temp = mean_heatmap[:, :, heatmap]
            bones[:, heatmap] = np.unravel_index(np.argmax(temp),[720,1280])
        
        
        if make_plot == True:
            sum_heatmap = np.sum(mean_heatmap, axis=2)/NUM_OF_JOINTS
            fig1 = plt.figure()
            ax1 = fig1.add_subplot(111)
            plot1 = ax1.imshow(sum_heatmap)
            text1 = 'heatmap max val:'+ str(np.max(sum_heatmap))
            plt.text(1,0.8,text1)
            plt.savefig(output_dir+'/heatmap' +str(num_of_photos) +'.png')
            plt.close()

            fig2 = plt.figure()
            ax2 = fig2.add_subplot(111)
            ax2.imshow(an_image)
            for j, bone in enumerate(bones_h36m):
                ax2.plot( bones[1,bone], bones[0,bone], color = 'w', linewidth=1)
            plt.savefig(output_dir+'/output' +str(num_of_photos) +'.png')
            plt.close()

        return bones
