from PythonClient import *
from AirSimClient import *
from math import cos, sin, pi, radians, degrees
import os, shutil
import time

SIZE_X = 1280
SIZE_Y = 720
FOCAL_LENGTH = SIZE_X/2
px = SIZE_X/2
py = SIZE_Y/2
CAMERA_OFFSET_X = 46/100
CAMERA_OFFSET_Y = 0
CAMERA_OFFSET_Z = 0
CAMERA_ROLL_OFFSET = 0
CAMERA_PITCH_OFFSET = -pi/4
CAMERA_YAW_OFFSET = 0
TORSO_SIZE = 0.395 #in meters


def EulerToRotationMatrix(roll, pitch, yaw):
    return np.array([[cos(yaw)*cos(pitch), cos(yaw)*sin(pitch)*sin(roll)-sin(yaw)*cos(roll), cos(yaw)*sin(pitch)*cos(roll)+sin(yaw)*sin(roll)],
                    [sin(yaw)*cos(pitch), sin(yaw)*sin(pitch)*sin(roll)+cos(yaw)*cos(roll), sin(yaw)*sin(pitch)*cos(roll)-cos(yaw)*sin(roll)],
                    [-sin(pitch), cos(pitch)*sin(roll), cos(pitch)*cos(roll)]])

R_cam = EulerToRotationMatrix (CAMERA_ROLL_OFFSET, -CAMERA_PITCH_OFFSET, CAMERA_YAW_OFFSET)
C_cam = np.array([[CAMERA_OFFSET_X, CAMERA_OFFSET_Y, -CAMERA_OFFSET_Z]]).T
FLIP_X_Y = np.array([[0,1,0],[-1,0,0],[0,0,1]])
FLIP_X_Y_inv = np.linalg.inv(FLIP_X_Y)

K = np.array([[FOCAL_LENGTH,0,px],[0,FOCAL_LENGTH,py],[0,0,1]])
K_inv = np.linalg.inv(K)

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

def TakeBoneProjection(P_world, R_drone, C_drone):
    P_drone = np.linalg.inv(np.vstack([np.hstack([R_drone, C_drone]), np.array([[0,0,0,1]])]))@np.vstack([P_world,  np.ones([1, P_world.shape[1]]) ] )
    P_camera =  np.linalg.inv(np.vstack([np.hstack([R_cam, C_cam]), np.array([[0,0,0,1]])]))@P_drone
    P_camera = P_camera[0:3,:]

    x = K@FLIP_X_Y@P_camera
    
    z = x[2,:]

    x[0,:] = x[0,:]/z
    x[1,:] = x[1,:]/z
    x = x[0:2, :]

    inFrame = True
    if np.any(x[0,:] < 0):
        inFrame = False
    if np.any(x[0,:] > SIZE_X):
        inFrame = False
    if np.any(x[1,:] < 0):
        inFrame = False
    if np.any(x[1,:] > SIZE_Y):
        inFrame = False

    return x, z, inFrame

def TakeBoneBackProjection(bone_pred, R_drone, C_drone, cov_, z_val, use_z = False):
    img_torso_size = np.linalg.norm(bone_pred[:, 0] - bone_pred[:, 8])
    calculated_z_val = (FOCAL_LENGTH * TORSO_SIZE) / img_torso_size

    if (use_z == False):
        z_val = calculated_z_val
    else:
        z_val = np.mean(z_val)

    bone_pos_3d = np.zeros([3, bone_pred.shape[1]])
    bone_pos_3d[0,:] = bone_pred[0,:]*z_val
    bone_pos_3d[1,:] = bone_pred[1,:]*z_val
    bone_pos_3d[2,:] = z_val
    
    bone_pos_3d = FLIP_X_Y_inv.dot(K_inv.dot(bone_pos_3d))

    P_camera = np.vstack([bone_pos_3d, np.ones([1,bone_pos_3d.shape[1]]) ])
    P_drone = np.hstack([R_cam, C_cam]).dot(P_camera)
    P_world = np.hstack([R_drone, C_drone]).dot(np.vstack([P_drone, np.ones([1, bone_pred.shape[1]])]))

    transformed_cov = (R_drone@R_cam)@cov_@(R_drone@R_cam).T

    return P_world, transformed_cov

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

