from PythonClient import *
from math import *
import time
#import numpy as np


#a small function to take photos and save them with name py[index].png to folder temp
def takePhoto(index):
    response = client.simGetImage(0, AirSimImageType.Scene)
    rawImage = np.fromstring(response, np.uint8)
    loc = 'temp/py' + str(index) + '.png'
    AirSimClient.write_file(os.path.normpath(loc), rawImage)
    return response

Z_POS = -0.7
DEGREE_ROTATION = 20 #amount of quantization
VELO = 3
NUM_OF_ROUNDS = 1 #how many times we want the drone to orbit hiker

#connect to the AirSim simulator
client = AirSimClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)
print('Taking off')
client.takeoff()
client.moveToPosition(0, 0, Z_POS, 0.75)

current_degree = 0
total_degree = 0
numOfPhotosTaken = 0

#find initial human and drone positions, and find the distance between them
droneLoc = client.getDroneWorldPosition()
humanLoc = client.getHumanPosition()
HUMAN_OFFSET = humanLoc
RADIUS =  sqrt((droneLoc.x_val-humanLoc.x_val)**2 + (droneLoc.y_val-humanLoc.y_val)**2) / 100
print ('Drone started ', RADIUS, 'm. from the hiker.\n')

path = []
while (float(total_degree) / 360 != NUM_OF_ROUNDS):
    #go to new position and face the hiker
    new_x = math.cos(math.radians(current_degree)) * RADIUS - RADIUS
    new_y = math.sin(math.radians(current_degree)) * RADIUS
    path.append(Vector3r(new_x, new_y, Z_POS))
    
    #update degree
    total_degree = DEGREE_ROTATION + total_degree
    current_degree = total_degree % 360
yaw_rate = (360-DEGREE_ROTATION)/(2*pi*RADIUS / VELO)
print(yaw_rate)
client.moveOnPath(path, VELO, 60, DrivetrainType.MaxDegreeOfFreedom, YawMode(True, yaw_rate))


print('End it!')


