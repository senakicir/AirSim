from AirSimClient import *
from math import *

import time
import cv2 as cv2
from helpers import *


def TakePhoto(index):
    response = client.simGetImages([ImageRequest(0, AirSimImageType.Scene)])
    response = response[0]
    bone_pos = client.getBonePositions()#response.bones
    loc = 'temp/img_' + str(index) + '.png'
    AirSimClient.write_file(os.path.normpath(loc), response.image_data_uint8)
    SaveBonePositions(num_of_photos, bone_pos, f_output)

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

filepath = 'temp/a_flight.txt'
f_output = open(filepath, 'w')

Z_POS = 6 #to do
BETA = 0.35
DELTA_T = 1
n = 5
time_horizon = n*DELTA_T

DRONE_POS_IND = 0
HUMAN_POS_IND = 1
R_SHOULDER_IND = 2
L_SHOULDER_IND = 3

#connect to the AirSim simulator
client = MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)
print('Taking off')
client.takeoff()

#find initial human and drone positions, and find the distance between them, find initial angle of drone
positions = client.getAllPositions() #everything is relative to drone initial drone pos now (drone is at the center of coord. system) = AirSim coord. system
shoulder_vector = positions[R_SHOULDER_IND, :] - positions[L_SHOULDER_IND, :] #find initial human orientation!
INITIAL_HUMAN_ORIENTATION = np.arctan2(-shoulder_vector[0], shoulder_vector[1]) #in unreal coordinates

projected_distance_vect = positions[HUMAN_POS_IND, :]
RADIUS =  np.linalg.norm(projected_distance_vect[0:2,]) #to do
print ('Drone started %.2f m. from the hiker.\n' % RADIUS)

drone_polar_pos = - positions[HUMAN_POS_IND, :] #find the drone initial orientation (needed for trackbar)
DRONE_ORIENTATION_OFFSET = np.arctan2(drone_polar_pos[1], drone_polar_pos[0])
DRONE_ORIENTATION_OFFSET = RangeAngle(DRONE_ORIENTATION_OFFSET, 360, True)
HUMAN_OFFSET = positions[HUMAN_POS_IND, :] #maybe i'll need this later

#define some variables
human_orientation = INITIAL_HUMAN_ORIENTATION
human_pos = positions[HUMAN_POS_IND, :] #wrt drone
drone_loc = np.array([0 ,0, 0])
linecount = 0

# create trackbars for angle change
cv2.namedWindow('Angle Control')
cv2.createTrackbar('Angle','Angle Control', 0, 360, doNothing)
cv2.setTrackbarPos('Angle', 'Angle Control', int(degrees(DRONE_ORIENTATION_OFFSET-INITIAL_HUMAN_ORIENTATION)))
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

    #get human position, delta human position, human drone_speedcity
    prev_human_pos = human_pos
    human_pos = positions[HUMAN_POS_IND,:]
    delta_human_pos = human_pos - prev_human_pos #how much the human moved in one iteration
    human_vel = delta_human_pos/DELTA_T #the velocity of the human (vector)
    human_speed = np.linalg.norm(human_vel) #the speed of the human (scalar)
    
    #what angle and polar position is the drone at currently
    drone_pos = client.getPosition() #airsim gives us the drone coordinates with initial drone loc. as origin
    drone_pos = np.array([drone_pos.x_val, drone_pos.y_val, drone_pos.z_val])
    current_polar_pos = (drone_pos - human_pos)     #subtrack the human_pos in order to find the current polar position vector.
    current_degree = np.arctan2(current_polar_pos[1], current_polar_pos[0]) #NOT relative to initial human angle

    #calculate human orientation
    shoulder_vector = positions[R_SHOULDER_IND, :] - positions[L_SHOULDER_IND, :]
    old_human_orientation = human_orientation
    #a filter to eliminate noisy data (smoother movement)
    human_orientation = np.arctan2(-shoulder_vector[0], shoulder_vector[1])*BETA + old_human_orientation*(1-BETA)
    human_rotation_speed = (human_orientation - old_human_orientation)/DELTA_T


    #calculate new polar coordinates according to circular motion (the circular offset required to rotate around human)
    input_rad = radians(cv2.getTrackbarPos('Angle', 'Angle Control')) #according to what degree we want the drone to be at
    input_rad_unreal_orient = input_rad + INITIAL_HUMAN_ORIENTATION #we don't use this at all currently
    desired_polar_angle = human_orientation + input_rad_unreal_orient #+ human_rotation_speed*time_horizon
    desired_polar_pos = np.array([cos(desired_polar_angle) * RADIUS, sin(desired_polar_angle) * RADIUS, 0])
    desired_pos = desired_polar_pos + human_pos + time_horizon*human_vel - np.array([0,0,Z_POS])
    desired_yaw = desired_polar_angle - pi
    # return desired_pos, desired_yaw
    
    delta_pos = desired_pos - drone_pos #how much the drone will have to move for this iteration
    desired_vel = delta_pos/time_horizon #The polar velocity we need.

    #find delta t and predict human position
    drone_speed = np.linalg.norm(desired_vel) #human_vel+polar_vel)
    #predicted_human = human_pos + n*DELTA_T * human_vel

    #angle required to face the hiker
    angle = client.getPitchRollYaw()
    current_yaw = angle[2]
    rotation_amount = desired_yaw - current_yaw
    rotation_amount = RangeAngle(rotation_amount, 180, True) #in radians

    #update drone position
    curr_pos = drone_pos
    new_pos = desired_pos

    #move drone!
    damping_yaw_rate = 1/(pi)
    damping_speed = 1
    client.moveToPosition(new_pos[0], new_pos[1], new_pos[2], drone_speed*damping_speed, 0,
                          DrivetrainType.MaxDegreeOfFreedom, YawMode(is_rate=False, yaw_or_rate=degrees(rotation_amount)*damping_yaw_rate), lookahead=-1, adaptive_lookahead=0)

    end = time.time()
    elapsed_time = end - start
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
    
    mystr = str(linecount)+'\t'+str(current_radius) +'\t'+str(np.linalg.norm(human_vel))+'\t'+str(drone_speed)+'\t'+str(real_drone_speed)+'\t'+str(np.linalg.norm(new_pos - curr_pos))+'\t'+str(degrees(angle[0]))+'\t'+str(degrees(angle[1]))+'\t'+str(desired_pos[0])+'\t'+str(desired_pos[1])+'\t'+str(desired_pos[2])+'\t'+str(drone_pos[0])+'\t'+str(drone_pos[1])+'\t'+str(drone_pos[2])
    mystr = mystr+'\n'
    f_output.write(mystr)
    linecount = linecount + 1
    print('linecount', linecount, 'current_radius', current_radius)

print('End it!')

