from PythonClient import *
from math import *

import time
import cv2 as cv2

#a small function to take photos and save them with name py[index].png to folder temp
def takePhoto(index):
    response = client.simGetImage(0, AirSimImageType.Scene)
    rawImage = np.fromstring(response, np.uint8)
    loc = 'temp/py' + str(index) + '.png'
    AirSimClient.write_file(os.path.normpath(loc), rawImage)
    return response

Z_POS = -2.6
VELO = 1.2
WAIT_TIME = 0.5
DEGREE_ROTATION = 20 #amount of quantization
NUM_OF_ROUNDS = 3 #how many times we want the drone to orbit hiker

#connect to the AirSim simulator
client = AirSimClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)
print('Taking off')
client.takeoff()

current_degree = 0
total_degree = current_degree
numOfPhotosTaken = 0

#find initial human and drone positions, and find the distance between them
droneLoc = client.getDroneWorldPosition()
humanLoc = client.getHumanPosition()
HUMAN_OFFSET = humanLoc
RADIUS =  sqrt((droneLoc.x_val-humanLoc.x_val)**2 + (droneLoc.y_val-humanLoc.y_val)**2) / 100
print ('Drone started %.2f m. from the hiker.\n' % RADIUS)

#set up plot stuff
drone_human_distance = []
velocity_arr = []
human_pos_x = 0
human_pos_y = 0
polar_x = 0
polar_y = 0
while (len(drone_human_distance) != 160):
    #get human position, delta human position, human velocity
    humanLoc = client.getHumanPosition()
    prev_human_x = human_pos_x
    prev_human_y = human_pos_y
    human_pos_x = (humanLoc.x_val - HUMAN_OFFSET.x_val)/100
    human_pos_y = (humanLoc.y_val - HUMAN_OFFSET.y_val)/100
    human_pos_z = (humanLoc.z_val - HUMAN_OFFSET.z_val)/100
    delta_human_x = (human_pos_x - prev_human_x)
    delta_human_y = (human_pos_y - prev_human_y)
    human_vel_x = delta_human_x/WAIT_TIME
    human_vel_y = delta_human_y/WAIT_TIME

    #get current degree
    drone_pos = client.getPosition()
    x_pos_drone = (drone_pos.x_val - human_pos_x + RADIUS)
    y_pos_drone = (drone_pos.y_val - human_pos_y)
    current_degree = degrees(np.arctan2(y_pos_drone, x_pos_drone))
    if current_degree < 0:
        current_degree = current_degree + 360

    #calculate new coordinates according to circular motion (the circular offset required to rotate around human)
    prev_polar_x = polar_x
    prev_polar_y = polar_y
    polar_x = math.cos(math.radians(current_degree + DEGREE_ROTATION)) * RADIUS - RADIUS
    polar_y = math.sin(math.radians(current_degree + DEGREE_ROTATION)) * RADIUS
    delta_polar_x = polar_x - prev_polar_x
    delta_polar_y = polar_y - prev_polar_y
    new_z = Z_POS

    #find delta t and predit human pos.
    delta_t = (math.sqrt(((delta_polar_x+delta_human_x)**2 + (delta_polar_y+delta_human_y)**2 )))/VELO
    predicted_human_x = human_pos_x + delta_t*human_vel_x
    predicted_human_y = human_pos_y + delta_t*human_vel_y

    #angle required to face the hiker
    desiredAngle = current_degree + DEGREE_ROTATION + 180
    angle = client.getRollPitchYaw()
    rotation_amount = (desiredAngle - degrees(angle[2]))%360

    #update drone position
    new_x = polar_x + predicted_human_x
    new_y = polar_y + predicted_human_y
    new_z = new_z - human_pos_z
    client.moveToPosition(new_x, new_y, new_z, VELO, WAIT_TIME, DrivetrainType.MaxDegreeOfFreedom, YawMode(False, rotation_amount/pi), -1, 1)

    #update degree
    total_degree = current_degree + total_degree

    #lets see if we got farther 
    oldDroneLoc = droneLoc
    droneLoc = client.getDroneWorldPosition()
    humanLoc = client.getHumanPosition()
    drone_human_distance.append(sqrt((droneLoc.x_val-humanLoc.x_val)**2 + (droneLoc.y_val-humanLoc.y_val)**2) / 100)
    velocity_arr.append(sqrt((droneLoc.x_val - oldDroneLoc.x_val)**2 + (droneLoc.y_val - oldDroneLoc.y_val)**2 +  (droneLoc.z_val - oldDroneLoc.z_val)**2 ))

print(drone_human_distance)
print(velocity_arr)

print('End it!')


