from projective_transform_plot import *
from math import *
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def EulerToRotationMatrix(roll, pitch, yaw):
    return np.array([[cos(yaw)*cos(pitch), cos(yaw)*sin(pitch)*sin(roll)-sin(yaw)*cos(roll), cos(yaw)*sin(pitch)*cos(roll)+sin(yaw)*sin(roll)],
                    [sin(yaw)*cos(pitch), sin(yaw)*sin(pitch)*sin(roll)+cos(yaw)*cos(roll), sin(yaw)*sin(pitch)*cos(roll)-cos(yaw)*sin(roll)],
                    [-sin(pitch), cos(pitch)*sin(roll), cos(pitch)*cos(roll)]])


##read lines from file
filepath = '../../../Airsim/2017-11-26/airsim_rec2017-11-26-18-58-49.txt'
filepath2 = '../../../Airsim/2017-11-26/plots/groundtruth.txt'
filepath3 = '../../../Airsim/2017-11-26/plots'
#filepath = 'temp/file.txt'
#filepath2 = 'temp/plots/groundtruth.txt'
#filepath3 = 'temp/plots'

SIZE_X = 1280
SIZE_Y = 720
CAMERA_OFFSET_X = 56
CAMERA_OFFSET_Y = 12.5
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
for line in f:
    linecount = linecount + 1
    numbers = line.strip().split('\t')
    numbers = list(map(float, numbers))
    
    #take the drone position and orientation
    [roll, pitch, yaw] = numbers[4:7]
    [Cx, Cy, Cz] = numbers[1:4]

    ##take orientation info and find R, rotation matrix
    ##take translation info and find C
    R_drone = EulerToRotationMatrix (roll, pitch, yaw)
    R_drone = FLIP_X_Y@R_drone@(FLIP_X_Y.T)

    C_drone = np.array([[Cx, Cy, Cz]]).T
    C_drone = FLIP_X_Y@C_drone
    
    #plot everything
    #PlotDroneAndHuman(numbers, linecount, filepath3)

    ##find H matrix
    fx = SIZE_X/2
    fy = SIZE_X/2
    px = SIZE_X/2
    py = SIZE_Y/2

    K = np.array([[fx,0,px],[0,fy,py],[0,0,1]])

    ##Take projective transform
    X = numbers[7:]
    X = np.reshape(X, (-1, 3)).T
    X = FLIP_X_Y.dot(X)
    T = np.linalg.inv(R_drone@R_cam)@(X-R_drone@C_cam-C_drone)
    x = K@T

    X1 = x[0,:]/x[2,:]
    X2 = x[1,:]/x[2,:]
    x[0,:] = X1
    x[1,:] = X2
    x = x[0:2, :]

    PlotProjection(x, linecount, filepath3)
    SuperImposeOnImage(x, filepath3, linecount)
    
    x = x.flatten('F')
    mystr = ''
    for element in x:
        mystr = mystr + '\t' + str(element)
    mystr = mystr + '\n'
    f_output.write(mystr)

f.close()
f_output.close()


