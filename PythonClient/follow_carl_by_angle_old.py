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

filepath = 'temp/a_flight.txt'
f_output = open(filepath, 'w')

Z_POS = -3.5
drone_vel = 0
DELTA_T = 1
n = 3

#connect to the AirSim simulator
client = MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)
print('Taking off')
client.takeoff()

#find initial human and drone positions, and find the distance between them, find initial angle of drone
positions = client.getBonePositions()
drone_loc = positions.dronePos
human_loc = positions.humanPos
a_val = positions.right_arm[b'x_val'] - positions.left_arm[b'x_val']
b_val = positions.right_arm[b'y_val'] - positions.left_arm[b'y_val']
HUMAN_ORIENTATION_OFFSET = degrees(np.arctan2(-a_val, b_val))
HUMAN_OFFSET = human_loc
RADIUS =  sqrt((drone_loc[b'x_val']-human_loc[b'x_val'])**2 + (drone_loc[b'y_val']-human_loc[b'y_val'])**2) / 100
DRONE_OFFSET_X = cos(radians(HUMAN_ORIENTATION_OFFSET))*RADIUS
DRONE_OFFSET_Y = sin(radians(HUMAN_ORIENTATION_OFFSET))*RADIUS
angle = client.getPitchRollYaw()
YAW_OFFSET = degrees(angle[2])
print ('Drone started %.2f m. from the hiker.\n' % RADIUS)

#define some variables
human_orientation = 0
human_pos_x = 0
human_pos_y = 0
human_pos_z = 0
new_x = 0
new_y = 0
new_z = 0
linecount = 0
num_of_photos = 0

# create trackbars for angle change
cv2.namedWindow('Angle Control')
cv2.createTrackbar('Angle','Angle Control',0,360, doNothing)
f_output = open('temp/file.txt', 'w')
mystr = 'linecount\t'+'current_radius\t'+'human_vel\t'+'drone_vel\t'+'real_drone_vel\t'+'drone travel dist\t'+'pitch\t'+'roll'
mystr = mystr+'\n'
f_output.write(mystr)
while True:
    start = time.time()
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
    #get human position, delta human position, human drone_velcity
    positions = client.getBonePositions()
    human_loc = positions.humanPos
    prev_human_x = human_pos_x
    prev_human_y = human_pos_y
    prev_human_z = human_pos_z
    #subtract initial location of human from current location. We're taking the initial location as the origin of our coord. system.
    human_pos_x = (human_loc[b'x_val'] - HUMAN_OFFSET[b'x_val'])/100
    human_pos_y = (human_loc[b'y_val'] - HUMAN_OFFSET[b'y_val'])/100
    human_pos_z = (human_loc[b'z_val'] - HUMAN_OFFSET[b'z_val'])/100
    delta_human_x = (human_pos_x - prev_human_x) #how much the human moved in one iteration
    delta_human_y = (human_pos_y - prev_human_y)
    delta_human_z = (human_pos_z - prev_human_z)
    human_vel_x = delta_human_x/DELTA_T  #the velocity of the human
    human_vel_y = delta_human_y/DELTA_T
    human_vel_z = delta_human_z/DELTA_T
    human_vel = sqrt(human_vel_x**2 + human_vel_y **2 + human_vel_z **2)

    #what angle is the drone at currently
    drone_pos = client.getPosition()
    x_pos_drone = (drone_pos.x_val - human_pos_x + DRONE_OFFSET_X)
    y_pos_drone = (drone_pos.y_val - human_pos_y + DRONE_OFFSET_Y)
    current_degree = degrees(np.arctan2(y_pos_drone, x_pos_drone)) - HUMAN_ORIENTATION_OFFSET
    if current_degree < -180:
        current_degree = current_degree + 360
    elif current_degree > 180:
        current_degree = current_degree - 360

    #calculate human orientation
    a_val = positions.right_arm[b'x_val']-positions.left_arm[b'x_val']     #calculate the human orientation
    b_val =  positions.right_arm[b'y_val']-positions.left_arm[b'y_val']
    old_human_orientation = human_orientation
    human_orientation = (degrees(np.arctan2(-a_val, b_val))-HUMAN_ORIENTATION_OFFSET)*0.2 + old_human_orientation*0.8

    #calculate new polar coordinates according to circular motion (the circular offset required to rotate around human)
    input_degree = cv2.getTrackbarPos('Angle', 'Angle Control') #according to what degree we want the drone to be at
    polar_x = cos(radians(human_orientation + input_degree + HUMAN_ORIENTATION_OFFSET)) * RADIUS - DRONE_OFFSET_X
    polar_y = sin(radians(human_orientation + input_degree + HUMAN_ORIENTATION_OFFSET)) * RADIUS - DRONE_OFFSET_Y
    current_polar_x = cos(radians(current_degree)) * RADIUS - DRONE_OFFSET_X
    current_polar_y = sin(radians(current_degree)) * RADIUS - DRONE_OFFSET_Y
    delta_polar_x = polar_x - current_polar_x #how much the drone will have to move for this iteration
    delta_polar_y = polar_y - current_polar_y
    polar_vel_x = delta_polar_x/(n*DELTA_T) #The polar velocity we need.
    polar_vel_y = delta_polar_y/(n*DELTA_T)
    print('current deg: ', current_degree, 'where to go', human_orientation + input_degree)

    #find delta t and predict human position
    drone_vel = math.sqrt((human_vel_x+polar_vel_x)**2 + (human_vel_y+polar_vel_y)**2 + (human_vel_z)**2)
    predicted_human_x = human_pos_x + n*DELTA_T * human_vel_x
    predicted_human_y = human_pos_y + n*DELTA_T * human_vel_y
    predicted_human_z = human_pos_z + n*DELTA_T * human_vel_z
    
    #angle required to face the hiker
    desiredAngle = human_orientation + input_degree
    angle = client.getPitchRollYaw()
    rotationAmount = (desiredAngle - (degrees(angle[2])-YAW_OFFSET))%360
    if rotationAmount > 180:
        rotationAmount = rotationAmount - 360
    elif rotationAmount < 180:
        rotationAmount = rotationAmount + 360


    #update drone position
    curr_x = drone_pos.x_val
    curr_y = drone_pos.y_val
    curr_z = drone_pos.z_val
    new_x = polar_x + predicted_human_x
    new_y = polar_y + predicted_human_y
    new_z = Z_POS - predicted_human_z
    
    #move drone!
    client.moveToPosition(new_x, new_y, new_z, drone_vel, 0, DrivetrainType.MaxDegreeOfFreedom, YawMode(False, rotationAmount/pi), -1, 0)

    end = time.time()
    elapsed_time = end - start
    if DELTA_T - elapsed_time > 0:
        time.sleep(DELTA_T - elapsed_time)

    #lets see if we got farther (mostly plot stuff)
    positions = client.getBonePositions()
    olddrone_loc = drone_loc
    drone_loc = positions.dronePos
    human_loc = positions.humanPos
    current_radius = sqrt((drone_loc[b'x_val']-human_loc[b'x_val'])**2 + (drone_loc[b'y_val']-human_loc[b'y_val'])**2) / 100
    real_drone_vel = sqrt(((drone_loc[b'x_val'] - olddrone_loc[b'x_val'])/DELTA_T)**2 + ((drone_loc[b'y_val'] - olddrone_loc[b'y_val'])/DELTA_T)**2 +  ((drone_loc[b'z_val'] - olddrone_loc[b'z_val'])/DELTA_T)**2 )/100

    mystr = str(linecount)+'\t'+str(current_radius) +'\t'+str(human_vel)+'\t'+str(drone_vel)+'\t'+str(real_drone_vel)+'\t'+str(sqrt((new_x-curr_x)**2+(new_y-curr_y)**2+ (new_z-curr_z)**2))+'\t'+str(degrees(angle[0]))+'\t'+str(degrees(angle[1]))
    mystr = mystr+'\n'
    f_output.write(mystr)
    linecount = linecount + 1
    print(linecount, 'rad: ', current_radius)

print('End it!')

