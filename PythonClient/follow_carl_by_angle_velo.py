from PythonClient import *
from math import *

import time
import cv2 as cv2

#a small function to take photos and save them with name py[index].png to folder temp
def TakePhoto(index):
    response = client.simGetImage(0, AirSimImageType.Scene)
    rawImage = np.fromstring(response, np.uint8)
    loc = 'temp/py' + str(index) + '.png'
    AirSimClient.write_file(os.path.normpath(loc), rawImage)
    return response

def doNothing(x):
    pass

Z_POS = -2
DELTA_T = 1

#connect to the AirSim simulator
client = AirSimClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)
print('Taking off')
client.takeoff()

#find initial human and drone positions, and find the distance between them
client.moveToPosition(0,0,Z_POS, 2, 1)
drone_loc = client.getDroneWorldPosition()
human_loc = client.getHumanPosition()
HUMAN_OFFSET = human_loc
RADIUS =  sqrt((drone_loc.x_val-human_loc.x_val)**2 + (drone_loc.y_val-human_loc.y_val)**2) / 100
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
real_drone_vel = 100

# create trackbars for angle change
cv2.namedWindow('Angle Control')
cv2.createTrackbar('Angle','Angle Control',0,360, doNothing)
times = 0
current_radius = RADIUS
while True:
    times = times +1
    start = time.time()
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
    #get human position, delta human position, human drone_velcity
    human_loc = client.getHumanPosition()
    prev_human_x = human_pos_x
    prev_human_y = human_pos_y
    prev_human_z = human_pos_z
    #subtract initial location of human from current location. We're taking the initial location as the origin of our coord. system.
    human_pos_x = (human_loc.x_val - HUMAN_OFFSET.x_val)/100
    human_pos_y = (human_loc.y_val - HUMAN_OFFSET.y_val)/100
    human_pos_z = (human_loc.z_val - HUMAN_OFFSET.z_val)/100
    delta_human_x = (human_pos_x - prev_human_x) #how much the human moved in one iteration
    delta_human_y = (human_pos_y - prev_human_y)
    delta_human_z = (human_pos_z - prev_human_z)
    human_vel_x = delta_human_x/DELTA_T  #the velocity of the human
    human_vel_y = delta_human_y/DELTA_T
    human_vel_z = delta_human_z/DELTA_T
    human_vel = sqrt(human_vel_x**2 + human_vel_y **2 + human_vel_z **2)

    #what angle is the drone at currently
    drone_pos = client.getPosition()
    x_pos_drone = (drone_pos.x_val - human_pos_x + RADIUS)
    y_pos_drone = (drone_pos.y_val - human_pos_y)
    current_degree = degrees(np.arctan2(y_pos_drone, x_pos_drone))
    if current_degree < 0:
        current_degree = current_degree + 360
    if current_degree > 360:
        current_degree = current_degree - 360
    
    #calculate new polar coordinates according to circular motion (the circular offset required to rotate around human)
    input_degree = cv2.getTrackbarPos('Angle', 'Angle Control') #according to what degree we want the drone to be at
    polar_x = math.cos(math.radians(input_degree)) * RADIUS - RADIUS
    polar_y = math.sin(math.radians(input_degree)) * RADIUS
    current_polar_x = math.cos(math.radians(current_degree)) * RADIUS - RADIUS
    current_polar_y = math.sin(math.radians(current_degree)) * RADIUS
    delta_polar_x = polar_x - current_polar_x #how much the drone will have to move for this iteration
    delta_polar_y = polar_y - current_polar_y
    n = 5
    polar_vel_x = delta_polar_x/(n*DELTA_T) #The polar velocity we need.
    polar_vel_y = delta_polar_y/(n*DELTA_T)
    new_z = Z_POS
    
    #find delta t and predict human position
    drone_vel_x = human_vel_x+polar_vel_x
    drone_vel_y = human_vel_y+polar_vel_y
    drone_vel_z = -human_vel_z
    drone_vel = math.sqrt((human_vel_x+polar_vel_x)**2 + (human_vel_y+polar_vel_y)**2 + human_vel_z**2)

    #angle required to face the hiker
    desiredAngle = input_degree + 180
    angle = client.getRollPitchYaw()
    rotationAmount = (desiredAngle - degrees(angle[2]))%360
    if rotationAmount > 180:
        rotationAmount = rotationAmount - 360
    #print(rotationAmount)
    
    #move drone!
    client.moveByVelocity(drone_vel_x, drone_vel_y, drone_vel_z, DELTA_T, drivetrain = DrivetrainType.MaxDegreeOfFreedom, yaw_mode =  YawMode(False, rotationAmount/pi))
    elapsed_time = time.time() - start
    if DELTA_T - elapsed_time > 0:
        time.sleep(DELTA_T - elapsed_time)

    #lets see if we got farther (mostly plot stuff)
    olddrone_loc = drone_loc
    drone_loc = client.getDroneWorldPosition()
    human_loc = client.getHumanPosition()
    current_radius = sqrt((drone_loc.x_val-human_loc.x_val)**2 + (drone_loc.y_val-human_loc.y_val)**2) / 100
    drone_human_distance.append(current_radius)
    real_drone_vel = sqrt((drone_loc.x_val - olddrone_loc.x_val)**2 + (drone_loc.y_val - olddrone_loc.y_val)**2 +  (drone_loc.z_val - olddrone_loc.z_val)**2 )/100
    drone_velcity_arr.append(real_drone_vel)

    print(current_radius, human_vel, drone_vel, real_drone_vel)


print(drone_human_distance)
print(drone_velcity_arr)

print('End it!')


