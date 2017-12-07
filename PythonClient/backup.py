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

Z_POS = -3.2 #to do
drone_vel = 0
BETA = 0.35
DELTA_T = 1
n = 3

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
unreal_positions = client.getAllPositions()
HUMAN_OFFSET = unreal_positions[HUMAN_POS_IND, :] #Unreal coordinates
positions = (unreal_positions - HUMAN_OFFSET)/100 #everything is relative to human now (human is at the center of coord. system)
shoulder_vector = positions[R_SHOULDER_IND, :] - positions[L_SHOULDER_IND, :] #find initial human orientation!
HUMAN_ORIENTATION_OFFSET = np.arctan2(-shoulder_vector[0], shoulder_vector[1]) #in unreal coordinates

projected_distance_vect = positions[DRONE_POS_IND, :]
RADIUS =  np.linalg.norm(projected_distance_vect[0:2,]) #to do
print ('Drone started %.2f m. from the hiker.\n' % RADIUS)

drone_loc = positions[DRONE_POS_IND, :]
DRONE_OFFSET = np.zeros((2,))
DRONE_ORIENTATION_OFFSET = np.arctan2(drone_loc[1], drone_loc[0])
DRONE_ORIENTATION_OFFSET = RangeAngle(DRONE_ORIENTATION_OFFSET, 360, True)
DRONE_OFFSET[0] = cos(DRONE_ORIENTATION_OFFSET)*RADIUS #where the drone starts centered around human, but according to original orientation
DRONE_OFFSET[1] = sin(DRONE_ORIENTATION_OFFSET)*RADIUS

#define some variables
human_orientation = HUMAN_ORIENTATION_OFFSET
human_pos = np.zeros((3,))
linecount = 0

# create trackbars for angle change
cv2.namedWindow('Angle Control')
cv2.createTrackbar('Angle','Angle Control',0,360, doNothing)
cv2.setTrackbarPos('Angle', 'Angle Control', int(degrees(DRONE_ORIENTATION_OFFSET-HUMAN_ORIENTATION_OFFSET)))
f_output = open('temp/file.txt', 'w')
mystr = 'linecount\t'+'current_radius\t'+'human_vel\t'+'drone_vel\t'+'real_drone_vel\t'+'drone travel dist\t'+'pitch\t'+'roll'
mystr = mystr+'\n'
f_output.write(mystr)
while True:
    start = time.time()
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
    
    #store unreal coordinate positions
    unreal_positions = client.getAllPositions()
    positions = (unreal_positions - HUMAN_OFFSET)/100 #translations are relative to initial human pos now (human is at the center of coord. system)
    
    #get human position, delta human position, human drone_velcity
    prev_human_pos = human_pos
    human_pos = positions[HUMAN_POS_IND,:]
    delta_human_pos = human_pos - prev_human_pos #how much the human moved in one iteration
    human_vel = delta_human_pos/DELTA_T #the velocity of the human (vector)
    human_speed = np.linalg.norm(human_vel) #the speed of the human (scalar)
    
    #what angle is the drone at currently
    drone_pos = client.getPosition() #airsim gives us the drone coordinates with initial drone loc. as origin
    #adding drone offset brings position from drone centered coord. frame to human centered coord. frame.
    #subtrack the human_pos in order to find the position vector.
    x_pos_drone = (drone_pos.x_val + DRONE_OFFSET[0] - human_pos[0] )
    y_pos_drone = (drone_pos.y_val + DRONE_OFFSET[1] - human_pos[1] )
    #calculate current degree of the drone by taking the arctangent of this position vector
    current_degree = np.arctan2(y_pos_drone, x_pos_drone) #NOT relative to initial human angle
    
    #calculate human orientation
    shoulder_vector = positions[R_SHOULDER_IND, :] - positions[L_SHOULDER_IND, :]
    old_human_orientation = human_orientation
    #a filter to eliminate noisy data (smoother movement)
    human_orientation = degrees(np.arctan2(-shoulder_vector[0], shoulder_vector[1]))*BETA + old_human_orientation*(1-BETA)
    #print('human_orientation', human_orientation)
    
    #calculate new polar coordinates according to circular motion (the circular offset required to rotate around human)
    input_degree = cv2.getTrackbarPos('Angle', 'Angle Control') #according to what degree we want the drone to be at
    input_degree_unreal_orient = input_degree + HUMAN_ORIENTATION_OFFSET
    polar_pos = np.array([cos(radians(human_orientation + input_degree_unreal_orient)) * RADIUS - DRONE_OFFSET[0],
                          sin(radians(human_orientation + input_degree_unreal_orient)) * RADIUS - DRONE_OFFSET[1], 0])
    current_polar_pos = np.array([cos(current_degree) * RADIUS - DRONE_OFFSET[0],
                                  sin(current_degree) * RADIUS - DRONE_OFFSET[1], 0])
    delta_polar_pos = polar_pos - current_polar_pos #how much the drone will have to move for this iteration
    polar_vel = delta_polar_pos/(n*DELTA_T) #The polar velocity we need.

    #find delta t and predict human position
    drone_vel = np.linalg.norm(human_vel+polar_vel)
    predicted_human = human_pos + n*DELTA_T * human_vel

    #angle required to face the hiker
    angle = client.getPitchRollYaw()
    current_yaw = degrees(angle[2]+pi) #relative to initial human orientation
    desired_yaw = human_orientation + input_degree_unreal_orient #relative to initial human orientation
    rotation_amount = desired_yaw - current_yaw
    rotation_amount = RangeAngle(rotation_amount, 180, False) #in radians

    #update drone position
    curr_pos = np.array([drone_pos.x_val, drone_pos.y_val, drone_pos.z_val])
    new_pos = np.array([polar_pos[0] + predicted_human[0], polar_pos[1] + predicted_human[1], Z_POS - predicted_human[2]])

    #move drone!
    client.moveToPosition(new_pos[0], new_pos[1], new_pos[2], drone_vel, 0, DrivetrainType.MaxDegreeOfFreedom, YawMode(False, rotation_amount/pi), -1, 0)

    end = time.time()
    elapsed_time = end - start
    if DELTA_T - elapsed_time > 0:
        time.sleep(DELTA_T - elapsed_time)

    #lets see if we got farther (mostly plot stuff)
    unreal_positions = client.getAllPositions()
    positions = (unreal_positions - HUMAN_OFFSET)/100 #everything is relative to human now (human is at the center of coord. system)
    olddrone_loc = drone_loc
    drone_loc = positions[DRONE_POS_IND,:]
    human_loc = positions[HUMAN_POS_IND,:]
    projected_distance_vect = drone_loc - human_loc
    current_radius =  np.linalg.norm(projected_distance_vect[0:2,])
    real_drone_vel = np.linalg.norm(drone_loc - olddrone_loc)
    
    mystr = str(linecount)+'\t'+str(current_radius) +'\t'+str(human_vel)+'\t'+str(drone_vel)+'\t'+str(real_drone_vel)+'\t'+str(np.linalg.norm(new_pos - curr_pos))+'\t'+str(degrees(angle[0]))+'\t'+str(degrees(angle[1]))
    mystr = mystr+'\n'
    f_output.write(mystr)
    linecount = linecount + 1
    print('linecount', linecount, 'current_radius', current_radius)

print('End it!')

