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
    if not os.path.exists('temp_main'):
        os.makedirs('temp_main')

    if not os.path.exists('temp_main/' + folder_name):
        os.makedirs('temp_main/' + folder_name)

    if not os.path.exists('temp_main/' + folder_name + '/estimates'):
        os.makedirs('temp_main/' + folder_name + '/estimates')

    if not os.path.exists('temp_main/' + folder_name + '/images'):
        os.makedirs('temp_main/' + folder_name + '/images')

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
    plt.title(str(errors["error_ave_pos"]))
    plt.savefig('temp_main/' + datetime_folder_name + '/estimates/est_vel_final' + '.png', bbox_inches='tight', pad_inches=0)
    plt.close()
    #################