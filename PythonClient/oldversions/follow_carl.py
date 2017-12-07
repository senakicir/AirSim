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
drone_vel = 0.0001
WAIT_TIME = 1

#connect to the AirSim simulator
client = AirSimClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)
print('Taking off')
client.takeoff()

#find initial human and drone positions, and find the distance between them
droneLoc = client.getDroneWorldPosition()
humanLoc = client.getHumanPosition()
HUMAN_OFFSET = humanLoc
RADIUS =  sqrt((droneLoc.x_val-humanLoc.x_val)**2 + (droneLoc.y_val-humanLoc.y_val)**2) / 100
print ('Drone started %.2f m. from the hiker.\n' % RADIUS)

#set up plot stuff
drone_human_distance = []
drone_velcity_arr = []

human_pos_x = 0
human_pos_y = 0
human_pos_z = 0
new_x = 0
new_y = 0
new_z = 0
human_vel = 0
while True:
    #get human position, delta human position, human drone_velcity
    humanLoc = client.getHumanPosition()
    prev_human_x = human_pos_x
    prev_human_y = human_pos_y
    prev_human_z = human_pos_z
    human_pos_x = (humanLoc.x_val - HUMAN_OFFSET.x_val)/100
    human_pos_y = (humanLoc.y_val - HUMAN_OFFSET.y_val)/100
    human_pos_z = (humanLoc.z_val - HUMAN_OFFSET.z_val)/100
    delta_human_x = (human_pos_x - prev_human_x)
    delta_human_y = (human_pos_y - prev_human_y)
    delta_human_z = (human_pos_z - prev_human_z)
    human_vel_x = delta_human_x/WAIT_TIME
    human_vel_y = delta_human_y/WAIT_TIME
    human_vel_z = delta_human_z/WAIT_TIME
    prev_human_vel = human_vel
    human_vel = math.sqrt(human_vel_x**2 + human_vel_y**2 + human_vel_z**2)

    #find delta t and predict human position
    delta_t = (math.sqrt(((delta_human_x)**2 + (delta_human_y)**2 )))/drone_vel
    predicted_human_x = human_pos_x + delta_t*human_vel_x
    predicted_human_y = human_pos_y + delta_t*human_vel_y

    #what angle am i at
    drone_pos = client.getPosition()
    x_pos_drone = (drone_pos.x_val - human_pos_x + RADIUS)
    y_pos_drone = (drone_pos.y_val - human_pos_y)
    current_degree = degrees(np.arctan2(y_pos_drone, x_pos_drone))
    if current_degree < 0:
        current_degree = current_degree + 360

    #angle required to face the hiker
    desiredAngle = current_degree + 180
    angle = client.getRollPitchYaw()
    rotation_amount = (desiredAngle - degrees(angle[2]))%360
    rotation_amount = 0

    #update drone position
    prev_x = new_x
    prev_y = new_y
    prev_z = new_z
    new_x = predicted_human_x
    new_y = predicted_human_y
    new_z = Z_POS - human_pos_z
    delta_movement = math.sqrt((prev_x-new_x)**2+(prev_y-new_y)**2+(prev_z-new_z)**2)
    
    #tune drone_velocity
    if (delta_movement > 0.5):
        client.moveToPosition(new_x, new_y, new_z, drone_vel, 0, DrivetrainType.MaxDegreeOfFreedom, YawMode(False, rotation_amount/pi), -1, 1)
        drone_vel = human_vel*0.80 + drone_vel*0.20
        #drone_vel = human_vel
        if drone_vel == 0:
            drone_vel = 0.001
    time.sleep(WAIT_TIME)

    #lets see if we got farther
    oldDroneLoc = droneLoc
    droneLoc = client.getDroneWorldPosition()
    humanLoc = client.getHumanPosition()
    drone_human_distance.append(sqrt((droneLoc.x_val-humanLoc.x_val)**2 + (droneLoc.y_val-humanLoc.y_val)**2) / 100)
    drone_velcity_arr.append(sqrt((droneLoc.x_val - oldDroneLoc.x_val)**2 + (droneLoc.y_val - oldDroneLoc.y_val)**2 +  (droneLoc.z_val - oldDroneLoc.z_val)**2 ))

    #orientation
    orientation = client.getDroneWorldOrientation()
    print(orientation.x_val, orientation.y_val, orientation.z_val)
    

print(drone_human_distance)
print(drone_velcity_arr)

print('End it!')


