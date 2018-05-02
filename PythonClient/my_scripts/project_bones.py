from helpers import *
import torch
from torch.autograd import Variable


SIZE_X = 1280
SIZE_Y = 720
FOCAL_LENGTH = SIZE_X/2
px = SIZE_X/2
py = SIZE_Y/2
CAMERA_OFFSET_X = 45/100
CAMERA_OFFSET_Y = 0
CAMERA_OFFSET_Z = 0
CAMERA_ROLL_OFFSET = 0
CAMERA_PITCH_OFFSET = -pi/4
CAMERA_YAW_OFFSET = 0
TORSO_SIZE = 0.424 #in meters
num_of_bones = 21
ones_tensor = Variable(torch.ones([1, num_of_bones]), requires_grad=False)*1.0
neat_tensor = Variable(torch.FloatTensor([[0, 0, 0, 1]]), requires_grad=False)

def euler_to_rotation_matrix(roll, pitch, yaw, returnTensor=False):
    if (returnTensor == True):
        return torch.FloatTensor([[cos(yaw)*cos(pitch), cos(yaw)*sin(pitch)*sin(roll)-sin(yaw)*cos(roll), cos(yaw)*sin(pitch)*cos(roll)+sin(yaw)*sin(roll)],
                    [sin(yaw)*cos(pitch), sin(yaw)*sin(pitch)*sin(roll)+cos(yaw)*cos(roll), sin(yaw)*sin(pitch)*cos(roll)-cos(yaw)*sin(roll)],
                    [-sin(pitch), cos(pitch)*sin(roll), cos(pitch)*cos(roll)]])
    return np.array([[cos(yaw)*cos(pitch), cos(yaw)*sin(pitch)*sin(roll)-sin(yaw)*cos(roll), cos(yaw)*sin(pitch)*cos(roll)+sin(yaw)*sin(roll)],
                    [sin(yaw)*cos(pitch), sin(yaw)*sin(pitch)*sin(roll)+cos(yaw)*cos(roll), sin(yaw)*sin(pitch)*cos(roll)-cos(yaw)*sin(roll)],
                    [-sin(pitch), cos(pitch)*sin(roll), cos(pitch)*cos(roll)]])

R_cam = euler_to_rotation_matrix (CAMERA_ROLL_OFFSET, -CAMERA_PITCH_OFFSET, CAMERA_YAW_OFFSET, returnTensor = False)
C_cam = np.array([[CAMERA_OFFSET_X, CAMERA_OFFSET_Y, CAMERA_OFFSET_Z]]).T
R_cam_torch = Variable(torch.from_numpy(R_cam).float(), requires_grad = False)
C_cam_torch = Variable(torch.FloatTensor([[CAMERA_OFFSET_X], [CAMERA_OFFSET_Y], [CAMERA_OFFSET_Z]]), requires_grad = False)
FLIP_X_Y = np.array([[0,1,0],[-1,0,0],[0,0,1]])
FLIP_X_Y_torch = Variable(torch.FloatTensor([[0,1,0],[-1,0,0],[0,0,1]]), requires_grad = False)
FLIP_X_Y_inv = np.linalg.inv(FLIP_X_Y)
FLIP_X_Y_inv_torch = torch.inverse(FLIP_X_Y_torch)

K = np.array([[FOCAL_LENGTH,0,px],[0,FOCAL_LENGTH,py],[0,0,1]])
K_torch = Variable(torch.FloatTensor([[FOCAL_LENGTH,0,px],[0,FOCAL_LENGTH,py],[0,0,1]]), requires_grad = False)
K_inv = np.linalg.inv(K)
K_inv_torch = torch.inverse(K_torch)

def take_bone_projection(P_world, R_drone, C_drone):

    P_drone = np.linalg.inv(np.vstack([np.hstack([R_drone, C_drone]), np.array([[0,0,0,1]])]))@np.vstack([P_world,  np.ones([1, P_world.shape[1]]) ] )
    P_camera =  np.linalg.inv(np.vstack([np.hstack([R_cam, C_cam]), np.array([[0,0,0,1]])]))@P_drone
    P_camera = P_camera[0:3,:]

    x_ = K@FLIP_X_Y@P_camera
    x = np.copy(x_)

    z = x[2,:]

    x[0,:] = x[0,:]/z
    x[1,:] = x[1,:]/z
    x = x[0:2, :]

    inFrame = True
    #if np.any(x[0,:] < 0):
    #    inFrame = False
    #if np.any(x[0,:] > SIZE_X):
    #    inFrame = False
    #if np.any(x[1,:] < 0):
    #    inFrame = False
    #if np.any(x[1,:] > SIZE_Y):
    #    inFrame = False

    return x, z, inFrame

def take_bone_projection_pytorch(P_world, R_drone, C_drone):    
    P_drone = torch.mm(torch.inverse(torch.cat((torch.cat((R_drone, C_drone), 1), neat_tensor), 0) ), torch.cat((P_world, ones_tensor), 0) )
    P_camera = torch.mm(torch.inverse(torch.cat((torch.cat((R_cam_torch, C_cam_torch), 1), neat_tensor), 0) ), P_drone)
    P_camera = P_camera[0:3,:]

    x_ = torch.mm(torch.mm(K_torch, FLIP_X_Y_torch), P_camera)
    x = x_.clone()
    
    z = x[2,:]

    result = Variable(torch.zeros([2,num_of_bones]), requires_grad = False)
    result[0,:] = x[0,:]/z
    result[1,:] = x[1,:]/z
    
    return result, z


def take_bone_backprojection(bone_pred, R_drone, C_drone, z_val, use_z = False):
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
    P_world_ = np.hstack([R_drone, C_drone]).dot(np.vstack([P_drone, np.ones([1, bone_pred.shape[1]])]))

    P_world = np.copy(P_world_)

    return P_world

def take_bone_backprojection_pytorch(bone_pred, R_drone, C_drone, z_val, use_z = False):
    img_torso_size = torch.norm(bone_pred[:, 0] - bone_pred[:, 8])
    calculated_z_val = (FOCAL_LENGTH * TORSO_SIZE) / img_torso_size

    if (use_z == False):
        z_val = calculated_z_val
    else:
        z_val = torch.mean(z_val)

    bone_pos_3d = Variable(torch.zeros([3, 21]))
    bone_pos_3d[0,:] = bone_pred[0,:]*z_val
    bone_pos_3d[1,:] = bone_pred[1,:]*z_val
    bone_pos_3d[2,:] = ones_tensor*z_val
    
    bone_pos_3d = torch.mm(FLIP_X_Y_inv_torch, torch.mm(K_inv_torch, bone_pos_3d))

    P_camera = torch.cat((bone_pos_3d, ones_tensor),0)
    P_drone = torch.mm(torch.cat((R_cam_torch, C_cam_torch),1), P_camera)
    P_world_ = torch.mm(torch.cat((R_drone, C_drone), 1) ,torch.cat((P_drone, ones_tensor),0))

    P_world = P_world_.clone()

    return P_world

def transform_cov_matrix(R_drone, cov_):
    transformed_cov = (R_drone@R_cam)@cov_@(R_drone@R_cam).T
    return transformed_cov
