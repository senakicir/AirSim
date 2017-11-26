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
WAIT_TIME = 8
DEGREE_ROTATION = 20 #amount of quantization
NUM_OF_ROUNDS = 2 #how many times we want the drone to orbit hiker

#connect to the AirSim simulator
client = AirSimClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)
print('Taking off')
client.takeoff()

current_degree = DEGREE_ROTATION
total_degree = current_degree
numOfPhotosTaken = 0

#find initial human and drone positions, and find the distance between them
droneLoc = client.getDroneWorldPosition()
humanLoc = client.getHumanPosition()
HUMAN_OFFSET = humanLoc
RADIUS =  sqrt((droneLoc.x_val-humanLoc.x_val)**2 + (droneLoc.y_val-humanLoc.y_val)**2) / 100
print ('Drone started %.2f m. from the hiker.\n' % RADIUS)

new_z = Z_POS
####delete yaw_rate = (360-DEGREE_ROTATION)/(2*pi*RADIUS / VELO)
#test bone function!
bonePos = client.getBonePositions()
print(bonePos.right_hand)

#set up kalman filter!
kalman = cv2.KalmanFilter(4,2)
kalman.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]],np.float32)
kalman.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]],np.float32)
kalman.processNoiseCov = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],np.float32) * 0.05
while (True):
    #angle required to face the hiker
    desiredAngle = current_degree + 180
    angle = client.getRollPitchYaw()
    rotation_amount = (desiredAngle - degrees(angle[2]))%360
    
    #keep track of distance between human and drone
    humanLoc = client.getHumanPosition()
    human_moved_x = (humanLoc.x_val - HUMAN_OFFSET.x_val)/100
    human_moved_y = (humanLoc.y_val - HUMAN_OFFSET.y_val)/100
    human_moved_z = (humanLoc.z_val - HUMAN_OFFSET.z_val)/100
    humanMeasurement = np.array([[np.float32(human_moved_x)],[np.float32(human_moved_y)]])
    
    #calculate coordinates according to circle
    new_x = math.cos(math.radians(current_degree)) * RADIUS - RADIUS
    new_y = math.sin(math.radians(current_degree)) * RADIUS
    new_z = Z_POS
    
    kalman.correct(humanMeasurement)
    predictedHumanPos = kalman.predict()
    new_x = new_x + human_moved_x#(int(predictedHumanPos[0]) / 100)
    new_y = new_y + human_moved_y#(int(predictedHumanPos[1]) / 100)
    new_z = new_z - human_moved_z
    print('%.4f ' %human_moved_z, '%.4f ' %new_z)
    
    client.moveToPosition(new_x, new_y, new_z, VELO, WAIT_TIME, DrivetrainType.MaxDegreeOfFreedom, YawMode(False, rotation_amount/pi), -1, 1)

    #update degree
    total_degree = DEGREE_ROTATION + total_degree
    current_degree = total_degree % 360

    #lets see if we got farther
    droneLoc = client.getDroneWorldPosition()
    humanLoc = client.getHumanPosition()
    drone_human_distance = sqrt((droneLoc.x_val-humanLoc.x_val)**2 + (droneLoc.y_val-humanLoc.y_val)**2) / 100
#    print(drone_human_distance, end=', ')

print('End it!')


