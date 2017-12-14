from AirSimClient import *
from math import *

import time
import cv2 as cv2
from helpers import *
import os, shutil

#constants
BETA = 0.35
DRONE_POS_IND = 0
HUMAN_POS_IND = 1
R_SHOULDER_IND = 2
L_SHOULDER_IND = 3
USE_TRACKBAR = True
INCREMENT_DEGREE_AMOUNT = radians(-45)
MAX_HUMAN_SPEED = 0.5


z_pos = 6 #to do
DELTA_T = 1
N = 3
TIME_HORIZON = N*DELTA_T

radius = 0
some_angle = 0

class State(object):
    def __init__(self, positions_):
        self.positions = positions_
        shoulder_vector = positions_[R_SHOULDER_IND, :] - positions_[L_SHOULDER_IND, :]
        self.human_orientation = np.arctan2(-shoulder_vector[0], shoulder_vector[1])
        self.human_rotation_speed = 0
        self.human_pos = positions_[HUMAN_POS_IND,:]
        self.human_vel = 0
        self.human_speed = 0
        self.drone_pos = np.array([0,0,0])
        self.current_polar_pos = np.array([0,0,0])
        self.current_degree = 0
        projected_distance_vect = positions_[HUMAN_POS_IND, :]
        global radius
        radius =  np.linalg.norm(projected_distance_vect[0:2,]) #to do
        drone_polar_pos = - positions_[HUMAN_POS_IND, :] #find the drone initial orientation (needed for trackbar)
        global some_angle
        some_angle = RangeAngle(np.arctan2(drone_polar_pos[1], drone_polar_pos[0]), 360, True)
    
    def updateState(self, positions_, client):
        self.positions = positions_
        
        #get human position, delta human position, human drone_speedcity
        prev_human_pos = self.human_pos
        self.human_pos = self.positions[HUMAN_POS_IND,:]
        delta_human_pos = self.human_pos - prev_human_pos #how much the human moved in one iteration
        self.human_vel = delta_human_pos/DELTA_T #the velocity of the human (vector)
        self.human_speed = np.linalg.norm(self.human_vel) #the speed of the human (scalar)
        
        #what angle and polar position is the drone at currently
        drone_pos_vec = client.getPosition() #airsim gives us the drone coordinates with initial drone loc. as origin
        self.drone_pos = np.array([drone_pos_vec.x_val, drone_pos_vec.y_val, drone_pos_vec.z_val])
        self.current_polar_pos = (self.drone_pos - self.human_pos)     #subtrack the human_pos in order to find the current polar position vector.
        self.current_degree = np.arctan2(self.current_polar_pos[1], self.current_polar_pos[0]) #NOT relative to initial human angle, not using currently
        print(self.current_degree)
        #calculate human orientation
        shoulder_vector = positions_[R_SHOULDER_IND, :] - positions_[L_SHOULDER_IND, :]
        prev_human_orientation = self.human_orientation
        #a filter to eliminate noisy data (smoother movement)
        self.human_orientation = np.arctan2(-shoulder_vector[0], shoulder_vector[1])*BETA + prev_human_orientation*(1-BETA)
        self.human_rotation_speed = (self.human_orientation-prev_human_orientation)/DELTA_T

def getDesiredPosAndAngle(state):
    desired_polar_angle = state.current_degree + INCREMENT_DEGREE_AMOUNT*(np.linalg.norm(state.human_vel)/MAX_HUMAN_SPEED)
    desired_polar_pos = np.array([cos(desired_polar_angle) * radius, sin(desired_polar_angle) * radius, 0])
    desired_pos = desired_polar_pos + state.human_pos + TIME_HORIZON*state.human_vel - np.array([0,0,z_pos])
    desired_yaw = desired_polar_angle - pi
    return desired_pos, desired_yaw

def getDesiredPosAndAngleTrackbar(state):
    #calculate new polar coordinates according to circular motion (the circular offset required to rotate around human)
    input_rad = radians(cv2.getTrackbarPos('Angle', 'Angle Control')) #according to what degree we want the drone to be at
    #input_rad_unreal_orient = input_rad + INITIAL_HUMAN_ORIENTATION #we don't use this at all currently
    desired_polar_angle = state.human_orientation + input_rad + state.human_rotation_speed*TIME_HORIZON
    desired_polar_pos = np.array([cos(desired_polar_angle) * radius, sin(desired_polar_angle) * radius, 0])
    desired_pos = desired_polar_pos + state.human_pos + TIME_HORIZON*state.human_vel - np.array([0,0,z_pos])
    desired_yaw = desired_polar_angle - pi
    return desired_pos, desired_yaw

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

def main():
    def TakePhoto(index):
        response = client.simGetImages([ImageRequest(0, AirSimImageType.Scene)])
        response = response[0]
        bone_pos = response.bones
        loc = 'temp/img_' + str(index) + '.png'
        AirSimClient.write_file(os.path.normpath(loc), response.image_data_uint8)
        SaveBonePositions2(num_of_photos, bone_pos, f_bones)
    
    for f in os.listdir('temp'):
        file_path = os.path.join('temp', f)
        if os.path.isfile(file_path):
            os.unlink(file_path)
        else:
            shutil.rmtree(file_path)
    f_output = open('temp/a_flight.txt', 'w')
    f_bones = open('temp/bones.txt', 'w')
    

    #connect to the AirSim simulator
    client = MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)
    print('Taking off')
    client.takeoff()

    #find initial human and drone positions, and find the distance between them, find initial angle of drone
    initial_positions = client.getAllPositions() #everything is relative to drone initial drone pos now (drone is at the center of coord. system) = AirSim coord. system
    current_state = State(initial_positions)

    shoulder_vector = initial_positions[R_SHOULDER_IND, :] - initial_positions[L_SHOULDER_IND, :] #find initial human orientation!
    INITIAL_HUMAN_ORIENTATION = np.arctan2(-shoulder_vector[0], shoulder_vector[1]) #in unreal coordinates

    print ('Drone started %.2f m. from the hiker.\n' % radius)

    #define some variables
    drone_loc = np.array([0 ,0, 0])
    linecount = 0
    num_of_photos = 0

    if (USE_TRACKBAR == True):
        # create trackbars for angle change
        cv2.namedWindow('Angle Control')
        cv2.createTrackbar('Angle','Angle Control', 0, 360, doNothing)
        cv2.setTrackbarPos('Angle', 'Angle Control', int(degrees(some_angle-INITIAL_HUMAN_ORIENTATION)))
    
    f_output = open('temp/file.txt', 'w')
    mystr = 'linecount\t'+'current_radius\t'+'human_vel\t'+'drone_speed\t'+'real_drone_speed\t'+'drone travel dist\t'+'pitch\t'+'roll\t'+'desired_x\t'+'desired_y\t'+'desired_z\t'+'drone_x\t'+'drone_y\t'+'drone_z\t'
    mystr = mystr+'\n'
    f_output.write(mystr)

    while True:
        start = time.time()
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
        
        # read unreal coordinate positions
        positions = client.getAllPositions() #translations are relative to initial drone pos now (drone is at the center of coord. system)
        current_state.updateState(positions, client) #updates human pos, human orientation, human vel, drone pos

        #finds desired position and angle
        if (USE_TRACKBAR == True):
            [desired_pos, desired_yaw] = getDesiredPosAndAngleTrackbar(current_state)
        else:
            [desired_pos, desired_yaw] = getDesiredPosAndAngle(current_state)
        
        #find desired drone speed
        delta_pos = desired_pos - current_state.drone_pos #how much the drone will have to move for this iteration
        desired_vel = delta_pos/TIME_HORIZON
        drone_speed = np.linalg.norm(desired_vel)

        #update drone position
        curr_pos = current_state.drone_pos
        new_pos = desired_pos

        #angle required to face the hiker
        angle = client.getPitchRollYaw()
        current_yaw = angle[2]
        rotation_amount = desired_yaw - current_yaw
        rotation_amount = RangeAngle(rotation_amount, 180, True) #in radians

        #move drone!
        damping_yaw_rate = 1/(pi)
        damping_speed = 1
        client.moveToPosition(new_pos[0], new_pos[1], new_pos[2], drone_speed*damping_speed, 0,
                              DrivetrainType.MaxDegreeOfFreedom, YawMode(is_rate=False, yaw_or_rate=degrees(rotation_amount)*damping_yaw_rate), lookahead=-1, adaptive_lookahead=0)

        end = time.time()
        elapsed_time = end - start
        
        TakePhoto(num_of_photos)
        num_of_photos = num_of_photos +1
        
        if DELTA_T - elapsed_time > 0:
            time.sleep(DELTA_T - elapsed_time)
        end = time.time()
        elapsed_time = end - start

        #lets see if we got farther (mostly plot stuff)
        positions = client.getAllPositions()
        olddrone_loc = drone_loc
        drone_loc = positions[DRONE_POS_IND,:]
        human_loc = positions[HUMAN_POS_IND,:]
        projected_distance_vect = drone_loc - human_loc
        current_radius =  np.linalg.norm(projected_distance_vect[0:2,])
        real_drone_speed = np.linalg.norm(drone_loc - olddrone_loc)/elapsed_time
        
        mystr = str(linecount)+'\t'+str(current_radius) +'\t'+str(np.linalg.norm(current_state.human_vel))+'\t'+str(drone_speed)+'\t'+str(real_drone_speed)+'\t'+str(np.linalg.norm(new_pos - curr_pos))+'\t'+str(degrees(angle[0]))+'\t'+str(degrees(angle[1]))+'\t'+str(desired_pos[0])+'\t'+str(desired_pos[1])+'\t'+str(desired_pos[2])+'\t'+str(current_state.drone_pos[0])+'\t'+str(current_state.drone_pos[1])+'\t'+str(current_state.drone_pos[2])
        mystr = mystr+'\n'
        f_output.write(mystr)
        linecount = linecount + 1
        print('linecount', linecount, 'current_radius', current_radius)

    print('End it!')

if __name__ == "__main__":
    main()

