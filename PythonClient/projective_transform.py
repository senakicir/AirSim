from projective_transform_plot import *
from math import *
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def EulerToRotationMatrix(roll, pitch, yaw):
    return np.array([[cos(yaw)*cos(pitch), cos(yaw)*sin(pitch)*sin(roll)-sin(yaw)*cos(roll), cos(yaw)*sin(pitch)*cos(roll)+sin(yaw)*sin(roll)],
                    [sin(yaw)*cos(pitch), sin(yaw)*sin(pitch)*sin(roll)+cos(yaw)*cos(roll), sin(yaw)*sin(pitch)*cos(roll)-cos(yaw)*sin(roll)],
                    [-sin(pitch), cos(pitch)*sin(roll), cos(pitch)*cos(roll)]])


def main():
    filepath = 'my_scripts/temp_main/test_set_2/groundtruth.txt'
    filepath2 = 'my_scripts/temp_main/test_set_2/groundtruth_projected.txt'
    filepath3 = 'my_scripts/temp_main/test_set_2/images/test_gt'
    filepath4 = 'my_scripts/temp_main/test_set_2/images'

    if not os.path.exists(filepath3):
        os.makedirs(filepath3)

    SIZE_X = 1280
    SIZE_Y = 720
    CAMERA_OFFSET_X = 46
    CAMERA_OFFSET_Y = 0
    CAMERA_OFFSET_Z = 0
    CAMERA_ROLL_OFFSET = 0
    CAMERA_PITCH_OFFSET = -pi/4
    CAMERA_YAW_OFFSET = 0
    FLIP_X_Y = np.array([[0,1,0],[-1,0,0],[0,0,1]])
    ##find H matrix
    FOCAL_LENGTH = SIZE_X/2
    px = SIZE_X/2
    py = SIZE_Y/2
    K = np.array([[FOCAL_LENGTH,0,px],[0,FOCAL_LENGTH,py],[0,0,1]])

    f_read  = open(filepath, 'r')
    f_output = open(filepath2, 'w')
    linecount = -1

    R_cam = EulerToRotationMatrix (CAMERA_ROLL_OFFSET, -CAMERA_PITCH_OFFSET, CAMERA_YAW_OFFSET)
    C_cam = np.array([[CAMERA_OFFSET_X, CAMERA_OFFSET_Y, -CAMERA_OFFSET_Z]]).T

    for line in f_read:
        linecount = linecount + 1
        print(linecount)
        numbers = line.strip().split('\t')
        numbers = list(map(float, numbers))
        
        #take the drone position and orientation
        [Cx, Cy, Cz] = numbers[1:4]
        [roll, pitch, yaw] = numbers[4:7]

        ##take orientation info and find R, rotation matrix
        ##take translation info and find C
        R_drone = EulerToRotationMatrix (roll, pitch, yaw)
        C_drone = np.array([[Cx, Cy, -Cz]]).T
        
        #plot everything
        photo_loc = filepath4 + '/img_'+ str(linecount)+'.png'
        #PlotDroneAndHuman(numbers, linecount, filepath3, photo_loc)

        ##Take projective transform
        X = numbers[10:]
        P_world = np.reshape(X, (-1, 3)).T
        P_world[2,:] = -P_world[2,:]


        P_drone = np.linalg.inv(np.vstack([np.hstack([R_drone, C_drone]), np.array([[0,0,0,1]])]))@np.vstack([P_world,  np.ones([1, P_world.shape[1]]) ] )
        P_camera =  np.linalg.inv(np.vstack([np.hstack([R_cam, C_cam]), np.array([[0,0,0,1]])]))@P_drone
        P_camera = P_camera[0:3,:]

        x = K@FLIP_X_Y@P_camera

        X1 = x[0,:]/x[2,:]
        X2 = x[1,:]/x[2,:]
        x[0,:] = X1
        x[1,:] = X2
        x = x[0:2, :]

        ######
        print(x[:,0], '\n')

        #PlotProjection(x, linecount, filepath3)
        SuperImposeOnImage(x, filepath3, linecount, photo_loc)
        x = x.flatten('F')
        mystr = str(linecount)
        for element in x:
            mystr = mystr + '\t' + str(element)
        mystr = mystr + '\n'
        f_output.write(mystr)


    f_read.close()
    f_output.close()

if __name__ == "__main__":
    main()
