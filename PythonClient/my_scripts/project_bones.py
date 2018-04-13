from helpers import *

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
TORSO_SIZE = 0.424 #in meters

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
