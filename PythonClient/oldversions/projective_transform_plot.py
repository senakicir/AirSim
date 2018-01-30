from math import *
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
from os import listdir

bones_h36m = [[0, 1], [1, 2], [2, 3], [3, 19],
              [0, 4], [4, 5], [5, 6], [6, 20],
              [0, 7], [7, 8], [8, 9], [9, 10],
              [8, 14], [14, 15], [15, 16], [16, 17],
              [8, 11], [11, 12], [12, 13], [13, 18]] #20 connections


joint_indices_h36m=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
joint_names_h36m = ['hip','right_up_leg','right_leg','right_foot','left_up_leg','left_leg', 'left_foot','spine1','neck','head', 'head_top', 'left_arm','left_forearm','left_hand','right_arm','right_forearm','right_hand', 'right_hand_tip', 'left_hand_tip', 'right_foot_tip', 'left_foot_tip']


colormap='gist_rainbow'

def PlotDroneAndHuman(numbers, linecount, location, photo_location):
    index = numbers[0]
    drone = numbers[1:4]
    drone = np.asarray(drone, dtype=np.float32)
    human = numbers[7:10]
    human = np.asarray(human, dtype=np.float32)
    joints = numbers[10:]
    joints = np.reshape(joints, (-1, 3)).T
    
    fig = plt.figure()
    gs1 = gridspec.GridSpec(1, 1)
    ax = fig.add_subplot(gs1[0], projection='3d')
    
    # maintain aspect ratio
    XYZ = np.hstack((drone[:, np.newaxis], joints))
    X = XYZ[0,:]
    Y = XYZ[1,:]
    Z = XYZ[2,:]
    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0
    mid_x = (X.max()+X.min()) * 0.5
    mid_y = (Y.max()+Y.min()) * 0.5
    mid_z = (Z.max()+Z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    #plot drone
    plot1 = ax.scatter(drone[0], drone[1], drone[2], c='r', marker='o')
    #plot joints
    for i, bone in enumerate(bones_h36m):
        plot2 = ax.plot(joints[0,bone], joints[1,bone], joints[2,bone], c='b', marker='^')

    text1 = 'human position:\n'+str(human[0])+'\n'+str(human[1])+'\n'+str(human[2])
    text2 = 'drone position:\n'+str(drone[0])+'\n'+str(drone[1])+'\n'+str(drone[2])

    plt.legend((plot1,plot1),(text1,text2), bbox_to_anchor=(1.6, 0.7))

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    lol = 'Image no: '+str(int(index))
    ax.set_title(lol)

    filename_photo = photo_location
    im = plt.imread(filename_photo)
    gs2 = gridspec.GridSpec(1, 1)
    ax2 = fig.add_subplot(gs2[0])
    ax2.imshow(im)


    gs1.tight_layout(fig, rect=[0, 0, 0.6, 0.6])
    gs2.tight_layout(fig, rect=[0.4, 0.4, 1, 1])
    filename = location + '/3d' + str(linecount) + '.png'

    plt.savefig(filename)
    plt.close()

def PlotProjection(numbers, linecount, location):

    fig = plt.figure()
    ax = fig.add_subplot(111)
    cmap = plt.get_cmap(colormap)
    colorindex = [17, 0, 5, 9, 15, 2, 18, 10, 12, 4, 14, 13, 11, 3, 7, 8, 16, 6, 1, 19]

    ax.set_xlim(0, 1280)
    ax.set_ylim(0, 720)
    ax.invert_yaxis()
    
    #you flipped axes here!
    for i, bone in enumerate(bones_h36m):
        color_ = cmap(colorindex[i]/len(bones_h36m))
        ax.plot(numbers[0,bone], numbers[1,bone], color = color_)

    filename = location + '/2d' + str(linecount) + '.png'
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()

def SuperImposeOnImage(numbers, location, linecount, photo_location):
    im = plt.imread(photo_location)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(im)
    
    #plot part
    cmap = plt.get_cmap(colormap)
    colorindex = [17, 0, 5, 9, 15, 2, 18, 10, 12, 4, 14, 13, 11, 3, 7, 8, 16, 6, 1, 19]
    for i, bone in enumerate(bones_h36m):
        color_ = cmap(colorindex[i]/len(bones_h36m))
        ax.plot( numbers[0,bone], numbers[1,bone], color = 'w', linewidth=1)

    filename = location + '/superimposed' + str(linecount) + '.png'
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()

