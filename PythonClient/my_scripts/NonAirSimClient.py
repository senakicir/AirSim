from AirSimClient import Vector3r_arr, Vector3r
import numpy as np 
import pandas as pd
from State import HUMAN_POS_IND, DRONE_POS_IND, DRONE_ORIENTATION_IND, L_SHOULDER_IND, R_SHOULDER_IND

class NonAirSimClient(object):
    def __init__(self, filename_bones, filename_others):
        self.groundtruth = pd.read_csv(filename_bones, sep='\t', header=None).ix[:,1:].as_matrix().astype('float')
        self.DRONE_INITIAL_POS = self.groundtruth[0,0:3]
        self.groundtruth = self.groundtruth[1:,:-1]
        self.a_flight = pd.read_csv(filename_others, sep='\t', header=None).ix
        self.a_flight = self.a_flight[:,1:].as_matrix().astype('float')
        self.linenumber = 0
        self.current_bone_pos = 0
        self.current_unreal_pos = 0
        self.current_drone_pos = 0
        self.current_drone_orient = 0
        self.num_of_data = self.groundtruth.shape[0]
        self.end = False

    def moveToPosition(self, arg1, arg2, arg3, arg4, arg5, arg6, arg7, yaw_or_rate=0 ,lookahead=0, adaptive_lookahead=0):
        self.linenumber = self.linenumber + 1
        if (self.linenumber == self.num_of_data-1):
            self.end = True

    def getPosition(self):
        position = Vector3r()
        (position.x_val, position.y_val, position.z_val) = self.a_flight[self.linenumber, 6:9]
        return position

    def getPitchRollYaw(self):
        (pitch, roll, yaw) = self.a_flight[self.linenumber, 3:6]
        return (pitch, roll, yaw)

    def updateSynchronizedData(self, unreal_positions_, bone_positions_, drone_pos_, drone_orient_):
        self.current_bone_pos = bone_positions_
        self.current_unreal_pos = unreal_positions_
        self.current_drone_pos = drone_pos_
        self.current_drone_orient = drone_orient_
        return 0
    
    def getSynchronizedData(self):
        return self.current_unreal_pos, self.current_bone_pos, self.current_drone_pos, self.current_drone_orient

    def simGetImages(self):
        response = DummyPhotoResponse()
        X = self.groundtruth[self.linenumber, :]
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
        return 0

    def changeAnimation(self, newAnim):
        return 0

class DummyPhotoResponse(object):
    bone_pos = np.array([])
    unreal_positions = np.zeros([5,3])
    image_data_uint8 = np.uint8(0)
