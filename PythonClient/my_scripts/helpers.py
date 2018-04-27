from PythonClient import *
from AirSimClient import *
from math import cos, sin, pi, radians, degrees
import os, shutil, skimage.io
import time
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def RangeAngle(angle, limit=360, is_radians = True):
    if is_radians == True:
        angle = degrees(angle)
    if angle > limit:
        angle = angle - 360
    elif angle < limit-360:
        angle = angle + 360
    if is_radians == True:
        angle = radians(angle)
    return angle

def SaveBonePositions2(index, bones, f_output):
    bones = [ v for v in bones.values() ]
    line = str(index)
    for i in range(0, len(bones)):
        line = line+'\t'+str(bones[i][b'x_val'])+'\t'+str(bones[i][b'y_val'])+'\t'+str(bones[i][b'z_val'])
    line = line+'\n'
    f_output.write(line)

def doNothing(x):
    pass

def resetAllFolders():
    folder_name = time.strftime("%Y-%m-%d-%H-%M")

    folder_names = ['temp_main', 'temp_main/' + folder_name, 'temp_main/' + folder_name + '/estimates', 'temp_main/' + folder_name + '/images', 'temp_main/' + folder_name + '/superimposed_images']
    for a_folder_name in folder_names:
        if not os.path.exists(a_folder_name):
            os.makedirs(a_folder_name)
    return folder_name

def plotErrorPlots(gt_hp_arr, est_hp_arr, gt_hv_arr, est_hv_arr, errors, datetime_folder_name):
    #PLOT STUFF HERE AT THE END OF SIMULATION
    fig1 = plt.figure()
    ax = fig1.add_subplot(111, projection='3d')
    ax.plot(gt_hp_arr[:, 0], gt_hp_arr[:, 1], gt_hp_arr[:, 2], c='b', marker='^')
    ax.plot(est_hp_arr[:, 0], est_hp_arr[:, 1], est_hp_arr[:, 2], c='r', marker='^')
    plt.title(str(errors["error_ave_pos"]))
    plt.savefig('temp_main/' + datetime_folder_name + '/estimates/est_pos_final' + '.png', bbox_inches='tight', pad_inches=0)
    plt.close()

    fig2 = plt.figure()
    ax = fig2.add_subplot(111, projection='3d')
    ax.plot(gt_hv_arr[:, 0], gt_hv_arr[:, 1], gt_hv_arr[:, 2], c='b', marker='^')
    ax.plot(est_hv_arr[:, 0], est_hv_arr[:, 1], est_hv_arr[:, 2], c='r', marker='^')
    plt.title(str(errors["error_ave_vel"]))
    plt.savefig('temp_main/' + datetime_folder_name + '/estimates/est_vel_final' + '.png', bbox_inches='tight', pad_inches=0)
    plt.close()
    #################


bones_h36m = [[0, 1], [1, 2], [2, 3], [3, 19],
              [0, 4], [4, 5], [5, 6], [6, 20],
              [0, 7], [7, 8], [8, 9], [9, 10],
              [8, 14], [14, 15], [15, 16], [16, 17],
              [8, 11], [11, 12], [12, 13], [13, 18]] #20 connections


joint_indices_h36m=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
joint_names_h36m = ['hip','right_up_leg','right_leg','right_foot','left_up_leg','left_leg', 'left_foot','spine1','neck','head', 'head_top', 'left_arm','left_forearm','left_hand','right_arm','right_forearm','right_hand', 'right_hand_tip', 'left_hand_tip', 'right_foot_tip', 'left_foot_tip']


colormap='gist_rainbow'
def SuperImposeOnImage(numbers, location, photo_location):
    im = plt.imread(photo_location)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(im)
    #plot part
    cmap = plt.get_cmap(colormap)
    colorindex = [17, 0, 5, 9, 15, 2, 18, 10, 12, 4, 14, 13, 11, 3, 7, 8, 16, 6, 1, 19]
    for i, bone in enumerate(bones_h36m):
        color_ = cmap(colorindex[i]/len(bones_h36m))
        ax.plot( numbers[0, bone], numbers[1,bone], color = 'b', linewidth=1)

    plt.savefig(location, bbox_inches='tight', pad_inches=0)
    plt.close()

def PlotDroneAndHuman(bones_GT, backprojected_bones, location):
    fig = plt.figure()
    gs1 = gridspec.GridSpec(1, 1)
    ax = fig.add_subplot(gs1[0], projection='3d')
    
    # maintain aspect ratio
    X = bones_GT[0,:]
    Y = bones_GT[1,:]
    Z = bones_GT[2,:]
    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() * 2
    mid_x = (X.max()+X.min()) * 0.5
    mid_y = (Y.max()+Y.min()) * 0.5
    mid_z = (Z.max()+Z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    #plot drone
    #plot1 = ax.scatter(drone[0], drone[1], drone[2], c='r', marker='o')
    #plot joints
    for i, bone in enumerate(bones_h36m):
        plot1 = ax.plot(bones_GT[0,bone], bones_GT[1,bone], bones_GT[2,bone], c='b', marker='^')
    for i, bone in enumerate(bones_h36m):
        plot2 = ax.plot(backprojected_bones[0,bone], backprojected_bones[1,bone], backprojected_bones[2,bone], c='r', marker='^')

    text1 = 'actual   :\n'+str(bones_GT[0])+'\n'+str(bones_GT[1])+'\n'+str(bones_GT[2])
    text2 = 'predicted:\n'+str(backprojected_bones[0])+'\n'+str(backprojected_bones[1])+'\n'+str(backprojected_bones[2])

    plt.legend((plot1,plot1),(text1,text2), bbox_to_anchor=(1.6, 0.7))

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.savefig(location)
    plt.close()
