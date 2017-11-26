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
WAIT_TIME = 1
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

new_z = Z_POS
bonePos = client.getBonePositions()
print(bonePos.right_hand)

#set up kalman filter!
kalman = cv2.KalmanFilter(4,4)
kalman.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],np.float32)
kalman.transitionMatrix = np.array([[1,0,WAIT_TIME,0],[0,1,0,WAIT_TIME],[0,0,1,0],[0,0,0,1]],np.float32)
kalman.processNoiseCov = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],np.float32) * 0.1
drone_human_distance = []
velocity_arr = []
human_moved_y = 0
human_moved_x = 0
while (len(drone_human_distance) != 160):
    #keep track of distance between human and drone
    humanLoc = client.getHumanPosition()
    prev_human_moved_x = human_moved_x
    human_moved_x = (humanLoc.x_val - HUMAN_OFFSET.x_val)/100
    prev_human_moved_y = human_moved_y
    human_moved_y = (humanLoc.y_val - HUMAN_OFFSET.y_val)/100
    human_moved_z = (humanLoc.z_val - HUMAN_OFFSET.z_val)/100
    vel_human_x = (human_moved_x-prev_human_moved_x)/WAIT_TIME
    vel_human_y = (human_moved_y-prev_human_moved_y)/WAIT_TIME
    humanMeasurement = np.array([[np.float32(human_moved_x)],[np.float32(human_moved_y)],[np.float32(vel_human_x)],[np.float32(vel_human_y)]])
    
    #what angle am i at
    drone_pos = client.getPosition()
    x_pos_drone = (drone_pos.x_val - human_moved_x + RADIUS)
    y_pos_drone = (drone_pos.y_val - human_moved_y)
                   
    current_degree = degrees(np.arctan2(y_pos_drone, x_pos_drone))
    if current_degree < 0:
        current_degree = current_degree + 360
    
    #calculate coordinates according to circle
    new_x = math.cos(math.radians(current_degree + DEGREE_ROTATION)) * RADIUS - RADIUS
    new_y = math.sin(math.radians(current_degree + DEGREE_ROTATION)) * RADIUS
    new_z = Z_POS
    
    #angle required to face the hiker
    desiredAngle = current_degree + DEGREE_ROTATION + 180
    angle = client.getRollPitchYaw()
    rotation_amount = (desiredAngle - degrees(angle[2]))%360
    
    kalman.correct(humanMeasurement)
    predictedHumanPos = kalman.predict()
    
    new_x = new_x + (int(predictedHumanPos[0]))
    new_y = new_y + (int(predictedHumanPos[1]))
    new_z = new_z - (human_moved_z)
    
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


