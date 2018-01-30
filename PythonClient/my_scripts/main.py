from human_2dpos import *
from helpers import *
from mpl_toolkits.mplot3d import Axes3D


#constants
BETA = 0.35
DRONE_POS_IND = 0
HUMAN_POS_IND = 1
R_SHOULDER_IND = 2
L_SHOULDER_IND = 3
DRONE_ORIENTATION_IND = 4

USE_TRACKBAR = False
INCREMENT_DEGREE_AMOUNT = radians(-45)
MAX_HUMAN_SPEED = 0.5
USE_GROUNDTRUTH = 1 #0 is groundtruth, 1 is mild-GT, 2 is real system

z_pos = 6 #to do
DELTA_T = 1
N = 3
TIME_HORIZON = N*DELTA_T

radius = 0
some_angle = 0
gt_hp = []
est_hp = []
count_est = 0

class State(object):
    def __init__(self, positions_):
        self.positions = positions_
        #shoulder_vector = positions_[R_SHOULDER_IND, :] - positions_[L_SHOULDER_IND, :]
        #self.human_orientation = np.arctan2(-shoulder_vector[0], shoulder_vector[1])
        #self.human_rotation_speed = 0
        self.human_pos = positions_[HUMAN_POS_IND,:]
        self.human_vel = 0
        self.human_speed = 0
        self.drone_pos = np.array([0,0,0])
        self.current_polar_pos = np.array([0,0,0])
        self.current_degree = 0
        self.drone_orientation = np.array([0,0,0])
        projected_distance_vect = positions_[HUMAN_POS_IND, :]

        global radius
        radius =  np.linalg.norm(projected_distance_vect[0:2,]) #to do

        drone_polar_pos = - positions_[HUMAN_POS_IND, :] #find the drone initial angle (needed for trackbar)
        global some_angle
        some_angle = RangeAngle(np.arctan2(drone_polar_pos[1], drone_polar_pos[0]), 360, True)
    
    def updateState(self, positions_):
        self.positions = positions_
        
        #get human position, delta human position, human drone_speedcity
        prev_human_pos = self.human_pos
        self.human_pos = self.positions[HUMAN_POS_IND,:]
        delta_human_pos = self.human_pos - prev_human_pos #how much the human moved in one iteration
        self.human_vel = delta_human_pos/DELTA_T #the velocity of the human (vector)
        self.human_speed = np.linalg.norm(self.human_vel) #the speed of the human (scalar)
        
        #what angle and polar position is the drone at currently
        self.drone_pos = positions_[DRONE_POS_IND, :] #airsim gives us the drone coordinates with initial drone loc. as origin
        self.drone_orientation = positions_[DRONE_ORIENTATION_IND, :]
        self.current_polar_pos = (self.drone_pos - self.human_pos)     #subtrack the human_pos in order to find the current polar position vector.
        self.current_degree = np.arctan2(self.current_polar_pos[1], self.current_polar_pos[0]) #NOT relative to initial human angle, not using currently
        #calculate human orientation
        #shoulder_vector = positions_[R_SHOULDER_IND, :] - positions_[L_SHOULDER_IND, :]
        #prev_human_orientation = self.human_orientation
        #a filter to eliminate noisy data (smoother movement)
        #self.human_orientation = np.arctan2(-shoulder_vector[0], shoulder_vector[1])*BETA + prev_human_orientation*(1-BETA)
        #self.human_rotation_speed = (self.human_orientation-prev_human_orientation)/DELTA_T

def getDesiredPosAndAngle(state):
    desired_polar_angle = state.current_degree + INCREMENT_DEGREE_AMOUNT*(np.linalg.norm(state.human_vel)/MAX_HUMAN_SPEED)
    desired_polar_pos = np.array([cos(desired_polar_angle) * radius, sin(desired_polar_angle) * radius, 0])
    desired_pos = desired_polar_pos + state.human_pos + TIME_HORIZON*state.human_vel - np.array([0,0,z_pos])
    desired_yaw = desired_polar_angle - pi
    return desired_pos, desired_yaw

def getDesiredPosAndAngleTrackbar(state):
    #calculate new polar coordinates according to circular motion (the circular offset required to rotate around human)
    input_rad = radians(cv2.getTrackbarPos('Angle', 'Drone Control')) #according to what degree we want the drone to be at
    current_radius = cv2.getTrackbarPos('Radius', 'Drone Control')
    desired_z_pos = cv2.getTrackbarPos('Z', 'Drone Control')
    #input_rad_unreal_orient = input_rad + INITIAL_HUMAN_ORIENTATION #we don't use this at all currently
    #desired_polar_angle = state.human_orientation + input_rad + state.human_rotation_speed*TIME_HORIZON
    desired_polar_angle = input_rad

    desired_polar_pos = np.array([cos(desired_polar_angle) * current_radius, sin(desired_polar_angle) * current_radius, 0])
    desired_pos = desired_polar_pos + state.human_pos + TIME_HORIZON*state.human_vel - np.array([0,0,desired_z_pos])
    desired_yaw = desired_polar_angle - pi
    return desired_pos, desired_yaw

def determineAllPositions(bone_pred, client):
    drone_pos_vec = client.getPosition() #airsim gives us the drone coordinates with initial drone loc. as origin
    angle = client.getPitchRollYaw()

    #TO DO;
    R_drone = EulerToRotationMatrix(angle[1], angle[0], angle[2])
    C_drone = np.array([drone_pos_vec.x_val, drone_pos_vec.y_val, drone_pos_vec.z_val]).T

    P_world, error = TakeBoneBackProjection(bone_pred, R_drone, C_drone, 0, False)

    positions = np.zeros([5, 3])    
    positions[DRONE_POS_IND,:] = np.array([drone_pos_vec.x_val, drone_pos_vec.y_val, drone_pos_vec.z_val])
    positions[DRONE_ORIENTATION_IND,:] = np.array([angle[0], angle[1], angle[2]])
    positions[HUMAN_POS_IND,:] = P_world[:,0]
    positions[HUMAN_POS_IND,2] = positions[HUMAN_POS_IND,2]+0.9

    return positions

def determineAllPositions_mildly_GT(bone_pos_GT, client):
    drone_pos_vec = client.getPosition() #airsim gives us the drone coordinates with initial drone loc. as origin
    unreal_positions = client.getAllPositions() 
    angle = client.getPitchRollYaw()

    R_drone = EulerToRotationMatrix(angle[1], angle[0], angle[2]) #pitch roll yaw
    C_drone = np.array([[drone_pos_vec.x_val, drone_pos_vec.y_val, drone_pos_vec.z_val]]).T

    bone_pred, z_val = TakeBoneProjection(bone_pos_GT, R_drone, C_drone)
    P_world, error = TakeBoneBackProjection(bone_pred, R_drone, C_drone, z_val, use_z = False)

    positions = np.zeros([5, 3])
    positions[DRONE_POS_IND,:] = np.array([drone_pos_vec.x_val, drone_pos_vec.y_val, drone_pos_vec.z_val])
    positions[DRONE_ORIENTATION_IND,:] = np.array([angle[0], angle[1], angle[2]])
    positions[HUMAN_POS_IND,:] = P_world[:,0]

    positions[HUMAN_POS_IND,2] = positions[HUMAN_POS_IND,2]+0.9

    gt_hp.append(unreal_positions[HUMAN_POS_IND, :])
    est_hp.append(positions[HUMAN_POS_IND, :])

    fig1 = plt.figure()
    ax = fig1.add_subplot(111, projection='3d')

    gt_hp_arr = np.asarray(gt_hp)
    est_hp_arr = np.asarray(est_hp)

    ax.plot(gt_hp_arr[:, 0], gt_hp_arr[:, 1], gt_hp_arr[:, 2], c='b', marker='^')
    ax.plot(est_hp_arr[:, 0], est_hp_arr[:, 1], est_hp_arr[:, 2], c='r', marker='^')
    global count_est
    plt.savefig('temp_main/estplot' + str(count_est) + '.png', bbox_inches='tight', pad_inches=0)
    plt.close()

    count_est = count_est + 1
    print(unreal_positions[HUMAN_POS_IND, :], positions[HUMAN_POS_IND, :], '\n')

    return positions

def determineAllPositions_all_GT(client, f_test):
    unreal_positions = client.getAllPositions() #airsim gives us the drone coordinates with initial drone loc. as origin
    angle = client.getPitchRollYaw()
    positions = np.zeros([5, 3])
    drone_pos_vec = client.getPosition()
    positions[DRONE_POS_IND,:] = np.array([drone_pos_vec.x_val, drone_pos_vec.y_val, drone_pos_vec.z_val])
    positions[HUMAN_POS_IND,:] = unreal_positions[HUMAN_POS_IND,:]
    positions[R_SHOULDER_IND,:] = unreal_positions[R_SHOULDER_IND,:]
    positions[L_SHOULDER_IND,:] = unreal_positions[L_SHOULDER_IND,:]
    
    positions[DRONE_ORIENTATION_IND,:] = np.array([angle[1], angle[0], angle[2]])

    my_str = str(positions[DRONE_POS_IND,0]) + '\t' + str(positions[DRONE_POS_IND,1]) + '\t' + str(positions[DRONE_POS_IND,2]) + '\t' + str(positions[DRONE_ORIENTATION_IND,0]) + '\t' + str(positions[DRONE_ORIENTATION_IND,1]) + '\t' + str(positions[DRONE_ORIENTATION_IND,2]) + '\t' + str(positions[HUMAN_POS_IND,0]) + '\t' + str(positions[HUMAN_POS_IND,1]) + '\t' + str(positions[HUMAN_POS_IND,2]) + '\n'
    f_test.write(my_str)
    return positions

def TakePhoto(client, index, f_bones):
    response = client.simGetImages([ImageRequest(0, AirSimImageType.Scene)])
    response = response[0]
    X = response.bones

    numbers = [ v for v in X.values() ]
    numbers = numbers[3:]
    bone_pos = np.zeros([3,len(numbers)])
    client.getAllPositions()
    unreal_drone_init_pos = client.DRONE_INITIAL_POS
    for i in range(0, len(numbers)):
        bone_pos[:,i] = np.array([numbers[i][b'x_val'], numbers[i][b'y_val'], -numbers[i][b'z_val']]) - unreal_drone_init_pos
    bone_pos = bone_pos / 100

    loc = 'temp_main/img_' + str(index) + '.png'
    AirSimClient.write_file(os.path.normpath(loc), response.image_data_uint8)
    #SaveBonePositions2(index, bone_pos, f_bones)
    return response.image_data_uint8, bone_pos

def main():
    
    if not os.path.exists('temp_main'):
        os.makedirs('temp_main')

    for f in os.listdir('temp_main'):
        file_path = os.path.join('temp_main', f)
        if os.path.isfile(file_path):
            os.unlink(file_path)
        else:
            shutil.rmtree(file_path)

    f_output = open('temp_main/a_flight.txt', 'w')
    f_bones = open('temp_main/bones.txt', 'w')
    f_test = open('temp_main/positions.txt', 'w')

    #connect to the AirSim simulator
    client = MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)
    print('Taking off')
    client.takeoff()



    #find initial human and drone positions, and find the distance between them, find initial angle of drone
    photo, bone_pos_GT = TakePhoto(client, 0, f_bones)

    if (USE_GROUNDTRUTH == 0):
        # read unreal coordinate positions
        initial_positions = determineAllPositions_all_GT(client, f_test)
    elif (USE_GROUNDTRUTH == 1):
        initial_positions = determineAllPositions_mildly_GT(bone_pos_GT, client)
    else:
        net = initNet()
        bone_pred = FindJointPos2D(photo, 0, net, make_plot = False)
        initial_positions = determineAllPositions(bone_pred, client) #translations are relative to initial drone pos now (drone is at the center of coord. system)

    current_state = State(initial_positions)

    #shoulder_vector = initial_positions[R_SHOULDER_IND, :] - initial_positions[L_SHOULDER_IND, :] #find initial human orientation!
    #INITIAL_HUMAN_ORIENTATION = np.arctan2(-shoulder_vector[0], shoulder_vector[1]) #in unreal coordinates

    print ('Drone started %.2f m. from the hiker.\n' % radius)

    #define some variables
    drone_loc = np.array([0 ,0, 0])
    linecount = 0
    num_of_photos = 1

    if (USE_TRACKBAR == True):
        # create trackbars for angle change
        cv2.namedWindow('Drone Control')
        cv2.createTrackbar('Angle','Drone Control', 0, 360, doNothing)
        #cv2.setTrackbarPos('Angle', 'Angle Control', int(degrees(some_angle-INITIAL_HUMAN_ORIENTATION)))
        cv2.setTrackbarPos('Angle', 'Drone Control', int(degrees(some_angle)))

        cv2.createTrackbar('Radius','Drone Control', 3, 10, doNothing)
        cv2.setTrackbarPos('Radius', 'Drone Control', int(radius))

        cv2.createTrackbar('Z','Drone Control', 3, 10, doNothing)
        cv2.setTrackbarPos('Z', 'Drone Control', z_pos)

    
    f_output = open('temp_main/file.txt', 'w')
    mystr = 'linecount\t'+'current_radius\t'+'human_vel\t'+'drone_speed\t'+'real_drone_speed\t'+'drone travel dist\t'+'pitch\t'+'roll\t'+'desired_x\t'+'desired_y\t'+'desired_z\t'+'drone_x\t'+'drone_y\t'+'drone_z\t'
    mystr = mystr+'\n'
    f_output.write(mystr)

    while True:

        start = time.time()
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

        photo, bone_pos_GT = TakePhoto(client, num_of_photos, f_bones)
        num_of_photos = num_of_photos +1

        if (USE_GROUNDTRUTH == 0):
            # read unreal coordinate positions
            positions = determineAllPositions_all_GT(client, f_test)
        elif (USE_GROUNDTRUTH == 1):
            positions = determineAllPositions_mildly_GT(bone_pos_GT, client)
        else:
            bone_pred = FindJointPos2D(photo, num_of_photos, net, make_plot = False)
            positions = determineAllPositions(bone_pred, client) #translations are relative to initial drone pos now (drone is at the center of coord. system)

        current_state.updateState(positions) #updates human pos, human orientation, human vel, drone pos

        #finds desired position and angle
        if (USE_TRACKBAR == True):
            [desired_pos, desired_yaw] = getDesiredPosAndAngleTrackbar(current_state)
        else:
            [desired_pos, desired_yaw] = getDesiredPosAndAngle(current_state)
        
        #find desired drone speed
        delta_pos = desired_pos - current_state.drone_pos #how much the drone will have to move for this iteration
        desired_vel = delta_pos/TIME_HORIZON
        drone_speed = np.linalg.norm(desired_vel)

        #update drone position
        curr_pos = current_state.drone_pos
        new_pos = desired_pos

        #angle required to face the hiker
        angle = current_state.drone_orientation
        current_yaw = angle[2]
        rotation_amount = desired_yaw - current_yaw
        rotation_amount = RangeAngle(rotation_amount, 180, True) #in radians

        #move drone!
        damping_yaw_rate = 1/(pi)
        damping_speed = 1
        client.moveToPosition(new_pos[0], new_pos[1], new_pos[2], drone_speed*damping_speed, 0,
                              DrivetrainType.MaxDegreeOfFreedom, YawMode(is_rate=False, yaw_or_rate=degrees(rotation_amount)*damping_yaw_rate), lookahead=-1, adaptive_lookahead=0)

        end = time.time()
        elapsed_time = end - start
        
        if DELTA_T - elapsed_time > 0:
            time.sleep(DELTA_T - elapsed_time)
        end = time.time()
        elapsed_time = end - start

        #lets see if we got farther (mostly plot stuff)
        #positions = client.getAllPositions()
        #olddrone_loc = drone_loc
        #drone_loc = positions[DRONE_POS_IND,:]
        #human_loc = positions[HUMAN_POS_IND,:]
        #projected_distance_vect = drone_loc - human_loc
        #current_radius =  np.linalg.norm(projected_distance_vect[0:2,])
        #real_drone_speed = np.linalg.norm(drone_loc - olddrone_loc)/elapsed_time
        
        #mystr = str(linecount)+'\t'+str(current_radius) +'\t'+str(np.linalg.norm(current_state.human_vel))+'\t'+str(drone_speed)+'\t'+str(real_drone_speed)+'\t'+str(np.linalg.norm(new_pos - curr_pos))+'\t'+str(degrees(angle[0]))+'\t'+str(degrees(angle[1]))+'\t'+str(desired_pos[0])+'\t'+str(desired_pos[1])+'\t'+str(desired_pos[2])+'\t'+str(current_state.drone_pos[0])+'\t'+str(current_state.drone_pos[1])+'\t'+str(current_state.drone_pos[2])
        #mystr = mystr+'\n'
        #f_output.write(mystr)
        #linecount = linecount + 1
        #print('linecount', linecount, 'current_radius', current_radius)

    print('End it!')

if __name__ == "__main__":
    main()

