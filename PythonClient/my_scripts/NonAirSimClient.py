from AirSimClient import Vector3r_arr, Vector3r
import numpy as np 
import pandas as pd
import torch
from State import HUMAN_POS_IND, DRONE_POS_IND, DRONE_ORIENTATION_IND, L_SHOULDER_IND, R_SHOULDER_IND

class NonAirSimClient(object):
    def __init__(self, filename_bones, filename_others):
        groundtruth_matrix = pd.read_csv(filename_bones, sep='\t', header=None).ix[:,1:].as_matrix().astype('float')                
        self.DRONE_INITIAL_POS = groundtruth_matrix[0,0:3]
        self.groundtruth = groundtruth_matrix[1:,:-1]
        a_flight_matrix = pd.read_csv(filename_others, sep='\t', header=None).ix
        self.a_flight = a_flight_matrix[:,1:].as_matrix().astype('float')
        self.linecount = 0
        self.current_bone_pos = 0
        self.current_unreal_pos = 0
        self.current_drone_pos = Vector3r()
        self.current_drone_orient = 0
        self.num_of_data = 10#self.a_flight.shape[0]
        self.error_2d = []
        self.error_3d = []
        self.requiredEstimationData = []
        self.naiveBackprojectionList = []
        self.end = False
        self.isCalibratingEnergy = True
        self.boneLengths = torch.zeros([20,1])


    def moveToPosition(self, arg1, arg2, arg3, arg4, arg5, arg6, arg7, yaw_or_rate=0 ,lookahead=0, adaptive_lookahead=0):
        if (self.linecount == self.num_of_data-1):
            self.end = True
        if (self.linecount == 5):
            self.isCalibratingEnergy = False

    def getPosition(self):
        position = Vector3r()
        (position.x_val, position.y_val, position.z_val) = self.a_flight[self.linecount, 6:9]
        return position

    def getPitchRollYaw(self):
        (pitch, roll, yaw) = self.a_flight[self.linecount, 3:6]
        return (pitch, roll, yaw)

    def updateSynchronizedData(self, unreal_positions_, bone_positions_, drone_pos_, drone_orient_):
        self.current_bone_pos = np.copy(bone_positions_)
        self.current_unreal_pos = np.copy(unreal_positions_)
        self.current_drone_pos = drone_pos_
        self.current_drone_orient = np.copy(drone_orient_)
        return 0
    
    def getSynchronizedData(self):
        return self.current_unreal_pos, self.current_bone_pos, self.current_drone_pos, self.current_drone_orient

    def simGetImages(self):
        response = DummyPhotoResponse()
        X = np.copy(self.groundtruth[self.linecount, :])
        line = np.reshape(X, (-1, 3))
        response.bone_pos = line[3:,:].T
        keys = {DRONE_POS_IND: 0, DRONE_ORIENTATION_IND: 1, HUMAN_POS_IND: 2}
        for key, value in keys.items():
            response.unreal_positions[key, :] = line[value, :] #dronepos
            if (key != DRONE_ORIENTATION_IND):
                response.unreal_positions[key, 2] = -response.unreal_positions[key, 2] #dronepos
                response.unreal_positions[key, :] = (response.unreal_positions[key, :] - self.DRONE_INITIAL_POS)/100
        response.bone_pos[2, :] = -response.bone_pos[2, :] 
        response.bone_pos = (response.bone_pos - self.DRONE_INITIAL_POS[:, np.newaxis])/100

        return response

    def reset(self):
        self.error_2d = []
        self.error_3d = []
        return 0

    def changeAnimation(self, newAnim):
        return 0

    def addNewFrame(self, pose_2d, R_drone, C_drone, pose3d_):
        self.requiredEstimationData.insert(0, [pose_2d, R_drone, C_drone])
        if (len(self.requiredEstimationData) > 6):
            self.requiredEstimationData.pop()

        self.naiveBackprojectionList.insert(0, pose3d_)
        if (len(self.naiveBackprojectionList) > 6):
            self.naiveBackprojectionList.pop()

class DummyPhotoResponse(object):
    bone_pos = np.array([])
    unreal_positions = np.zeros([5,3])
    image_data_uint8 = np.uint8(0)
