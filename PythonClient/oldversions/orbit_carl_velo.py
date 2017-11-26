from PythonClient import *
from math import *
import time
import cv2

#a small function to take photos and save them with name py[index].png to folder temp
def takePhoto(index):
    response = client.simGetImage(0, AirSimImageType.Scene)
    rawImage = np.fromstring(response, np.uint8)
    loc = 'temp/py' + str(index) + '.png'
    AirSimClient.write_file(os.path.normpath(loc), rawImage)
    return response

Z_POS = -0.7
DEGREE_ROTATION = 20 #amount of quantization
NUM_OF_ROUNDS = 1 #how many times we want the drone to orbit hiker
DURATION = 1

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

yaw_rate = (360-DEGREE_ROTATION)/(2*pi*RADIUS / 2)
vx = 0
vy = 0

while (float(total_degree) / 360 != NUM_OF_ROUNDS):
    #go to new position and face the hiker
    new_x = math.cos(math.radians(current_degree)) * RADIUS - RADIUS
    new_y = math.sin(math.radians(current_degree)) * RADIUS
    vx = (2*new_x / DURATION) - vx
    vy = (2*new_y / DURATION) - vy
    client.moveByVelocity(vx, vy, 0, DURATION, DrivetrainType.MaxDegreeOfFreedom, YawMode(False, current_degree - 180))
    time.sleep(DURATION)

    
    #update degree
    total_degree = DEGREE_ROTATION + total_degree
    current_degree = total_degree % 360
    
    #keep track of distance between human and drone
    droneLoc = client.getDroneWorldPosition()
    humanLoc = client.getHumanPosition()
#print('%.2f,' % (droneLoc.x_val-humanLoc.x_val), ' %.2f,' % (droneLoc.y_val-humanLoc.y_val), ' %.2f' % (droneLoc.z_val-humanLoc.z_val))



print('End it!')


