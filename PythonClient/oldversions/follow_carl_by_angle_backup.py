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
human_vel = 0
polar_x = 0
polar_y = 0


# create trackbars for angle change
cv2.namedWindow('Angle Control')
cv2.createTrackbar('Angle','Angle Control',0,360, doNothing)
while True:
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
    human_vel_x = delta_human_x/WAIT_TIME #the velocity of the human
    human_vel_y = delta_human_y/WAIT_TIME
    human_vel_z = delta_human_z/WAIT_TIME
    prev_human_vel = human_vel

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
#    delta_polar_x = polar_x - prev_polar_x #how much the drone will have to move for this iteration ##delete!!!!
#    delta_polar_y = polar_y - prev_polar_y
#    polar_vel_x = delta_polar_x/(1*WAIT_TIME) #The polar velocity we need.
#    polar_vel_y = delta_polar_y/(1*WAIT_TIME)
    new_z = Z_POS

    #find delta t and predict human position
    delta_t = (math.sqrt(((delta_polar_x + delta_human_x)**2 + (delta_polar_y + delta_human_y)**2 )))/drone_vel
    predicted_human_x = human_pos_x + delta_t*human_vel_x
    predicted_human_y = human_pos_y + delta_t*human_vel_y
    #add this polar velocity to the human velocity to get a nice velocity for the drone
    drone_vel = math.sqrt((human_vel_x+polar_vel_x)**2 + (human_vel_y+polar_vel_y)**2 + human_vel_z**2)
    if drone_vel == 0:
        drone_vel = 0.001

    #angle required to face the hiker
    desiredAngle = input_degree + 180
    angle = client.getRollPitchYaw()
    rotationAmount = (desiredAngle - degrees(angle[2]))%360
    if rotationAmount > 180:
        rotationAmount = rotationAmount - 360
    #print(rotationAmount)

    #update drone position
    prev_x = new_x
    prev_y = new_y
    prev_z = new_z
    new_x = polar_x + predicted_human_x
    new_y = polar_y + predicted_human_y
    new_z = new_z - human_pos_z
    delta_movement = math.sqrt((prev_x-new_x)**2+(prev_y-new_y)**2+(prev_z-new_z)**2)
    
    #tune drone_velocity
    if (delta_movement > 0):
        client.moveToPosition(new_x, new_y, new_z, drone_vel, 0, DrivetrainType.MaxDegreeOfFreedom, YawMode(False, rotationAmount/pi), -1, 1)
    time.sleep(WAIT_TIME)

    #lets see if we got farther
    olddrone_loc = drone_loc
    drone_loc = client.getDroneWorldPosition()
    human_loc = client.getHumanPosition()
    current_radius = sqrt((drone_loc.x_val-human_loc.x_val)**2 + (drone_loc.y_val-human_loc.y_val)**2) / 100
    drone_human_distance.append(current_radius)
    current_vel = sqrt((drone_loc.x_val - olddrone_loc.x_val)**2 + (drone_loc.y_val - olddrone_loc.y_val)**2 +  (drone_loc.z_val - olddrone_loc.z_val)**2 )
    drone_velcity_arr.append(current_vel)

    print( 'RADIUS: ', current_radius, delta_human_x, delta_human_y)
    

print(drone_human_distance)
print(drone_velcity_arr)

print('End it!')


