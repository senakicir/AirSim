import cv2 as cv2
from math import radians, cos, sin, pi
import numpy as np
from helpers import RangeAngle

#constants
BETA = 0.35
DRONE_POS_IND = 0
HUMAN_POS_IND = 1
R_SHOULDER_IND = 2
L_SHOULDER_IND = 3
DRONE_ORIENTATION_IND = 4

INCREMENT_DEGREE_AMOUNT = radians(-30)

z_pos = 6 
DELTA_T = 1
N = 3
TIME_HORIZON = N*DELTA_T

class State(object):
    def __init__(self, positions_):
        self.positions = positions_
        #shoulder_vector = positions_[R_SHOULDER_IND, :] - positions_[L_SHOULDER_IND, :]
        #self.human_orientation = np.arctan2(-shoulder_vector[0], shoulder_vector[1])
        #self.human_rotation_speed = 0
        self.human_pos = positions_[HUMAN_POS_IND,:]
        self.human_vel = 1. * np.array([0,0,0])
        self.human_speed = 0
        self.drone_pos = np.array([0,0,0])
        self.current_polar_pos = np.array([0,0,0])
        self.current_degree = 0
        self.drone_orientation = np.array([0,0,0])
        projected_distance_vect = positions_[HUMAN_POS_IND, :]
        self.inFrame = True
        self.radius = np.linalg.norm(projected_distance_vect[0:2,]) #to do

        drone_polar_pos = - positions_[HUMAN_POS_IND, :] #find the drone initial angle (needed for trackbar)
        self.some_angle = RangeAngle(np.arctan2(drone_polar_pos[1], drone_polar_pos[0]), 360, True)

        self.kalman = cv2.KalmanFilter(3, 3, 0) #3, state variables. 3 measurement variables 

        #3x3 F: no need for further modification
        self.kalman.transitionMatrix = 1. * np.eye(3,3)
        #3x3: H just takes the position values from state. 
        self.kalman.measurementMatrix = 1. * np.eye(3,3)
        #3x3: Process noise, no need to modify
        self.kalman.processNoiseCov = 1e-5 * np.eye(3,3)
        #3x3: measurement noise, WILL NEED TO MODIFY
        self.kalman.measurementNoiseCov = 1. * np.eye(3,3)#np.array([[1e-2, 0, 0], [0, 1e-2, 0], [0, 0, 1]])
        #3x3 initial covariance matrix
        self.kalman.errorCovPost = 1. * np.eye(3, 3)
        #3x1 initial state, no need to modify
        self.kalman.statePost = np.array([[self.human_pos[0], self.human_pos[1], self.human_pos[2]]]).T
        self.kalman.statePre = np.array([[self.human_pos[0], self.human_pos[1], self.human_pos[2]]]).T

        #3x1 control matrix, no need to modify
        self.kalman.controlMatrix = 1. * np.array([[DELTA_T, 0, 0], [0, DELTA_T, 0], [0, 0, DELTA_T]])
        
    
    def updateState(self, positions_, inFrame_, cov_):
        self.kalman.measurementNoiseCov = cov_
        self.positions = positions_
        self.inFrame = inFrame_
        #get human position, delta human position, human drone_speedcity
        prediction_human_pos = self.kalman.predict(self.human_vel)
        estimated_human_state = self.kalman.correct(self.positions[HUMAN_POS_IND,:])

        prev_human_pos = self.human_pos
        self.human_pos = np.copy(estimated_human_state[0:3,0])
        self.human_vel = (self.human_pos - prev_human_pos)/DELTA_T
        self.human_speed = np.linalg.norm(self.human_vel) #the speed of the human (scalar)
        
        #what angle and polar position is the drone at currently
        self.drone_pos = positions_[DRONE_POS_IND, :] #airsim gives us the drone coordinates with initial drone loc. as origin
        self.drone_orientation = positions_[DRONE_ORIENTATION_IND, :]
        self.current_polar_pos = (self.drone_pos - self.human_pos)     #subtrack the human_pos in order to find the current polar position vector.
        self.current_degree = np.arctan2(self.current_polar_pos[1], self.current_polar_pos[0]) #NOT relative to initial human angle, not using currently
        #calculate human orientation
        #shoulder_vector = positions_[R_SHOULDER_IND, :] - positions_[L_SHOULDER_IND, :]
        #prev_human_orientation = self.human_orientation
        #a filter to eliminate noisy data (smoother movement)
        #self.human_orientation = np.arctan2(-shoulder_vector[0], shoulder_vector[1])*BETA + prev_human_orientation*(1-BETA)
        #self.human_rotation_speed = (self.human_orientation-prev_human_orientation)/DELTA_T

    def getDesiredPosAndAngle(self):
        desired_polar_angle = self.current_degree + INCREMENT_DEGREE_AMOUNT
        desired_polar_pos = np.array([cos(desired_polar_angle) * self.radius, sin(desired_polar_angle) * self.radius, 0])
        desired_pos = desired_polar_pos + self.human_pos + TIME_HORIZON*self.human_vel - np.array([0,0,z_pos])
        #desired_pos = desired_polar_pos + self.human_pos - np.array([0,0,z_pos])
        desired_yaw = desired_polar_angle - pi
        return desired_pos, desired_yaw

    def getDesiredPosAndAngleTrackbar(self):
        #calculate new polar coordinates according to circular motion (the circular offset required to rotate around human)
        input_rad = radians(cv2.getTrackbarPos('Angle', 'Drone Control')) #according to what degree we want the drone to be at
        current_radius = cv2.getTrackbarPos('Radius', 'Drone Control')
        desired_z_pos = cv2.getTrackbarPos('Z', 'Drone Control')
        #input_rad_unreal_orient = input_rad + INITIAL_HUMAN_ORIENTATION #we don't use this at all currently
        #desired_polar_angle = state.human_orientation + input_rad + state.human_rotation_speed*TIME_HORIZON
        desired_polar_angle = input_rad

        desired_polar_pos = np.array([cos(desired_polar_angle) * current_radius, sin(desired_polar_angle) * current_radius, 0])
        #desired_pos = desired_polar_pos + self.human_pos + TIME_HORIZON*self.human_vel - np.array([0,0,desired_z_pos])
        desired_pos = desired_polar_pos + state.human_pos - np.array([0,0,desired_z_pos])
        desired_yaw = desired_polar_angle - pi
        return desired_pos, desired_yaw