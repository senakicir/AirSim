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
    ##read lines from file
    DATE = '2017-12-12-11-44-28'
    filepath = '../../../Airsim/'+DATE+'/airsim_rec.txt'
    filepath2 = '../../../Airsim/'+DATE+'/plots/groundtruth.txt'
    filepath3 = '../../../Airsim/'+DATE+'/plots'
    filepath4 = '../../../Airsim/'+DATE+'/images'
    
    filepath = 'temp/bones.txt'
    filepath2 = 'temp/plots/groundtruth.txt'
    filepath3 = 'temp/plots'
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
    FLIP_X_Y = np.array([[0,-1,0],[1,0,0],[0,0,1]])

    f  = open(filepath, 'r')
    f_output = open(filepath2, 'w')
    linecount = -1

    R_cam = EulerToRotationMatrix (CAMERA_ROLL_OFFSET, CAMERA_PITCH_OFFSET, CAMERA_YAW_OFFSET)
    R_cam = FLIP_X_Y@R_cam@(FLIP_X_Y.T)
    C_cam = np.array([[CAMERA_OFFSET_X, CAMERA_OFFSET_Y, CAMERA_OFFSET_Z]]).T
    C_cam = FLIP_X_Y.dot(C_cam)
    next(f)
    photoname_list = listdir(filepath4)
    for line in f:
        linecount = linecount + 1
        print(linecount)
        numbers = line.strip().split('\t')
        numbers = numbers[:-1]
        numbers = list(map(float, numbers))
        
        #take the drone position and orientation
        [roll, pitch, yaw] = numbers[4:7]
        [Cx, Cy, Cz] = numbers[1:4]

        ##take orientation info and find R, rotation matrix
        ##take translation info and find C
        R_drone = EulerToRotationMatrix (-roll, -pitch, yaw)
        R_drone = FLIP_X_Y@R_drone@(FLIP_X_Y.T)

        C_drone = np.array([[Cx, Cy, Cz]]).T
        C_drone = FLIP_X_Y@C_drone
        
        #plot everything
        photo_loc = filepath4 + '/'+ photoname_list[linecount]
        PlotDroneAndHuman(numbers, linecount, filepath3, photo_loc)

        ##find H matrix
        fx = SIZE_X/2
        fy = SIZE_X/2
        px = SIZE_X/2
        py = SIZE_Y/2

        K = np.array([[fx,0,px],[0,fy,py],[0,0,1]])

        ##Take projective transform
        X = numbers[10:]
        X = np.reshape(X, (-1, 3)).T
        X = FLIP_X_Y.dot(X)
        T = (np.linalg.inv(R_drone@R_cam))@(X-R_drone@C_cam-C_drone)
        x = K@T

        X1 = x[0,:]/x[2,:]
        X2 = x[1,:]/x[2,:]
        x[0,:] = X1
        x[1,:] = X2
        x = x[0:2, :]

        PlotProjection(x, linecount, filepath3)
        SuperImposeOnImage(x, filepath3, linecount, photo_loc)
        
        x = x.flatten('F')
        mystr = ''
        for element in x:
            mystr = mystr + '\t' + str(element)
        mystr = mystr + '\n'
        f_output.write(mystr)

    f.close()
    f_output.close()

if __name__ == "__main__":
    main()
