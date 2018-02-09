from AirSimClient import Vector3r_arr, Vector3r
import numpy as np 
import pandas as pd
from State import HUMAN_POS_IND

class NonAirSimClient(object):
    def __init__(self, filename_bones, filename_others):

        self.bones = pd.read_csv(filename_bones, sep='\t', header=None).ix[:,1:].as_matrix().astype('float')
        self.others = pd.read_csv(filename_others, sep='\t', header=None).ix
        self.others = self.others[:,1:].as_matrix().astype('float')
        self.linenumber = 0
        self.num_of_data = self.bones.shape[0]
        self.end = False

    def moveToPosition(self, arg1, arg2, arg3, arg4, arg5, arg6, arg7, yaw_or_rate=0 ,lookahead=0, adaptive_lookahead=0):
        self.linenumber = self.linenumber + 1
        if (self.linenumber == self.num_of_data-1):
            self.end = True

    def getPosition(self):
        position = Vector3r()
        (position.x_val, position.y_val, position.z_val) = self.others[self.linenumber, 6:9]
        return position

    def getPitchRollYaw(self):
        (pitch, roll, yaw) = self.others[self.linenumber, 3:6]
        return (pitch, roll, yaw)

    def getAllPositions(self):
        positionsArr = np.zeros((4,3))
        positionsArr[HUMAN_POS_IND,:] = self.others[self.linenumber, 0:3]
        return positionsArr

    def simGetImages(self):
        response = DummyPhotoResponse()
        X = self.bones[self.linenumber, :]
        response.bones  = np.reshape(X, (-1, 3)).T
        return response

class DummyPhotoResponse(object):
    bones = np.array([])
    image_data_uint8 = np.uint8(0)
