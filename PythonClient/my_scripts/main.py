from human_2dpos import *
from helpers import *
from State import *
from NonAirSimClient import *
from mpl_toolkits.mplot3d import Axes3D

USE_TRACKBAR = False
USE_GROUNDTRUTH = 1 #0 is groundtruth, 1 is mild-GT, 2 is real system
USE_AIRSIM = True

datetime_folder_name = ''
gt_hv = []
est_hv = []
mystr = ''

def determineAllPositions(bone_pred, client):
    drone_pos_vec = client.getPosition() #airsim gives us the drone coordinates with initial drone loc. as origin
    angle = client.getPitchRollYaw()

    #TO DO;
    R_drone = EulerToRotationMatrix(angle[1], angle[0], angle[2])
    C_drone = np.array([drone_pos_vec.x_val, drone_pos_vec.y_val, drone_pos_vec.z_val]).T

    P_world, _ = TakeBoneBackProjection(bone_pred, R_drone, C_drone, 0, False)

    positions = np.zeros([5, 3])    
    positions[DRONE_POS_IND,:] = np.array([drone_pos_vec.x_val, drone_pos_vec.y_val, drone_pos_vec.z_val])
    positions[DRONE_ORIENTATION_IND,:] = np.array([angle[0], angle[1], angle[2]])
    positions[HUMAN_POS_IND,:] = P_world[:,0]
    positions[HUMAN_POS_IND,2] = positions[HUMAN_POS_IND,2]+0.9

    return positions

def determineAllPositions_mildly_GT(bone_pos_GT, client):
    drone_pos_vec = client.getPosition() #airsim gives us the drone coordinates with initial drone loc. as origin
    angle = client.getPitchRollYaw()

    R_drone = EulerToRotationMatrix(angle[1], angle[0], angle[2]) #pitch roll yaw
    C_drone = np.array([[drone_pos_vec.x_val, drone_pos_vec.y_val, drone_pos_vec.z_val]]).T

    bone_pred, z_val, inFrame = TakeBoneProjection(bone_pos_GT, R_drone, C_drone)
    P_world, _, cov = TakeBoneBackProjection(bone_pred, R_drone, C_drone, z_val, use_z = False)

    positions = np.zeros([5, 3])
    positions[DRONE_POS_IND,:] = np.array([drone_pos_vec.x_val, drone_pos_vec.y_val, drone_pos_vec.z_val])
    positions[DRONE_ORIENTATION_IND,:] = np.array([angle[0], angle[1], angle[2]])
    positions[HUMAN_POS_IND,:] = P_world[:,0]
    positions[HUMAN_POS_IND,2] = positions[HUMAN_POS_IND,2]+0.9

    unreal_positions = client.getAllPositions() #airsim gives us the drone coordinates with initial drone loc. as origin

    global mystr
    mystr = ''
    mystr = '\t'+str(unreal_positions[HUMAN_POS_IND, 0]) +'\t'+str(unreal_positions[HUMAN_POS_IND, 1])+'\t'+str(unreal_positions[HUMAN_POS_IND, 2])+'\t'+str(angle[0])+'\t'+str(angle[1])+'\t'+str(angle[2])+'\t'+str(drone_pos_vec.x_val)+'\t'+str(drone_pos_vec.y_val)+'\t'+str(drone_pos_vec.z_val)
    mystr = mystr+'\n'
    return positions, unreal_positions, inFrame, cov

def determineAllPositions_all_GT(client):
    unreal_positions = client.getAllPositions() #airsim gives us the drone coordinates with initial drone loc. as origin
    angle = client.getPitchRollYaw()
    positions = np.zeros([5, 3])
    drone_pos_vec = client.getPosition()
    positions[DRONE_POS_IND,:] = np.array([drone_pos_vec.x_val, drone_pos_vec.y_val, drone_pos_vec.z_val])
    positions[HUMAN_POS_IND,:] = unreal_positions[HUMAN_POS_IND,:]
    positions[R_SHOULDER_IND,:] = unreal_positions[R_SHOULDER_IND,:]
    positions[L_SHOULDER_IND,:] = unreal_positions[L_SHOULDER_IND,:]
    
    positions[DRONE_ORIENTATION_IND,:] = np.array([angle[1], angle[0], angle[2]])

    global mystr
    mystr = ''
    mystr = '\t'+str(unreal_positions[HUMAN_POS_IND, 0]) +'\t'+str(unreal_positions[HUMAN_POS_IND, 1])+'\t'+str(unreal_positions[HUMAN_POS_IND, 2])+'\t'+str(angle[0])+'\t'+str(angle[1])+'\t'+str(angle[2])+'\t'+str(drone_pos_vec.x_val)+'\t'+str(drone_pos_vec.y_val)+'\t'+str(drone_pos_vec.z_val)
    mystr = mystr+'\n'
    return positions

def TakePhoto(client, index):
    if (USE_AIRSIM == True):
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
        
        loc = 'temp_main/' + datetime_folder_name + '/images/img_' + str(index) + '.png'
        AirSimClient.write_file(os.path.normpath(loc), response.image_data_uint8)
    else:
        response = client.simGetImages()
        bone_pos = response.bones

    return response.image_data_uint8, bone_pos

def main():
    
    end_test = False
    global datetime_folder_name
    datetime_folder_name = resetAllFolders()

    f_output = open('temp_main/' + datetime_folder_name + '/a_flight.txt', 'w')
    f_bones = open('temp_main/' + datetime_folder_name + '/bones.txt', 'w')

    #connect to the AirSim simulator
    if (USE_AIRSIM == True):
        client = MultirotorClient()
        client.confirmConnection()
        client.enableApiControl(True)
        client.armDisarm(True)
        print('Taking off')
        client.takeoff()
    else:
        filename_bones = 'temp_main/test_set_1/bones.txt'
        filename_output = 'temp_main/test_set_1/a_flight.txt'
        client = NonAirSimClient(filename_bones, filename_output)

    #find initial human and drone positions, and find the distance between them, find initial angle of drone
    photo, bone_pos_GT = TakePhoto(client, 0)

    if (USE_GROUNDTRUTH == 0):
        # read unreal coordinate positions
        initial_positions = determineAllPositions_all_GT(client)
    elif (USE_GROUNDTRUTH == 1):
        initial_positions, _, _, _ = determineAllPositions_mildly_GT(bone_pos_GT, client)
    else:
        net = initNet()
        bone_pred = FindJointPos2D(photo, 0, net, make_plot = False)
        initial_positions = determineAllPositions(bone_pred, client) #translations are relative to initial drone pos now (drone is at the center of coord. system)

    current_state = State(initial_positions)

    #shoulder_vector = initial_positions[R_SHOULDER_IND, :] - initial_positions[L_SHOULDER_IND, :] #find initial human orientation!
    #INITIAL_HUMAN_ORIENTATION = np.arctan2(-shoulder_vector[0], shoulder_vector[1]) #in unreal coordinates

    print ('Drone started %.2f m. from the hiker.\n' % current_state.radius)

    #define some variables
    drone_loc = np.array([0 ,0, 0])
    linecount = 0
    num_of_photos = 1
    count_est = 0
    gt_hp = []
    est_hp = []

    if (USE_TRACKBAR == True):
        # create trackbars for angle change
        cv2.namedWindow('Drone Control')
        cv2.createTrackbar('Angle','Drone Control', 0, 360, doNothing)
        #cv2.setTrackbarPos('Angle', 'Angle Control', int(degrees(some_angle-INITIAL_HUMAN_ORIENTATION)))
        cv2.setTrackbarPos('Angle', 'Drone Control', int(degrees(current_state.some_angle)))

        cv2.createTrackbar('Radius','Drone Control', 3, 10, doNothing)
        cv2.setTrackbarPos('Radius', 'Drone Control', int(current_state.radius))

        cv2.createTrackbar('Z','Drone Control', 3, 10, doNothing)
        cv2.setTrackbarPos('Z', 'Drone Control', z_pos)

    while (end_test == False):

        start = time.time()
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

        photo, bone_pos_GT = TakePhoto(client, num_of_photos)
        num_of_photos = num_of_photos +1

        if (USE_GROUNDTRUTH == 0):
            # read unreal coordinate positions
            positions = determineAllPositions_all_GT(client)
            inFrame = True
        elif (USE_GROUNDTRUTH == 1):
            positions, unreal_positions, inFrame, cov = determineAllPositions_mildly_GT(bone_pos_GT, client)
        else:
            inFrame = True #TO DO
            bone_pred = FindJointPos2D(photo, num_of_photos, net, make_plot = False)
            positions = determineAllPositions(bone_pred, client) #translations are relative to initial drone pos now (drone is at the center of coord. system)

        current_state.updateState(positions, inFrame, cov) #updates human pos, human orientation, human vel, drone pos
        
        #PLOT STUFF HERE#
        gt_hp.append(unreal_positions[HUMAN_POS_IND, :])
        est_hp.append(current_state.human_pos)
        gt_hp_arr = np.asarray(gt_hp)
        est_hp_arr = np.asarray(est_hp)

        fig1 = plt.figure()
        ax = fig1.add_subplot(111, projection='3d')
        ax.plot(gt_hp_arr[:, 0], gt_hp_arr[:, 1], gt_hp_arr[:, 2], c='b', marker='^')
        ax.plot(est_hp_arr[:, 0], est_hp_arr[:, 1], est_hp_arr[:, 2], c='r', marker='^')
        plt.savefig('temp_main/' + datetime_folder_name + '/estimates/est_pos' + str(count_est) + '.png', bbox_inches='tight', pad_inches=0)
        plt.close()

        if count_est > 0: 
            gt_hv.append((gt_hp_arr[count_est, :]-gt_hp_arr[count_est-1, :])/DELTA_T)
            est_hv.append(current_state.human_vel)
            gt_hv_arr = np.asarray(gt_hv)
            est_hv_arr = np.asarray(est_hv)

            fig2 = plt.figure()
            ax = fig2.add_subplot(111, projection='3d')
            ax.plot(gt_hv_arr[:, 0], gt_hv_arr[:, 1], gt_hv_arr[:, 2], c='b', marker='^')
            ax.plot(est_hv_arr[:, 0], est_hv_arr[:, 1], est_hv_arr[:, 2], c='r', marker='^')
            plt.savefig('temp_main/' + datetime_folder_name + '/estimates/est_vel' + str(count_est) + '.png', bbox_inches='tight', pad_inches=0)
            plt.close()

        count_est = count_est + 1
        #################

        #finds desired position and angle
        if (USE_TRACKBAR == True):
            [desired_pos, desired_yaw] = current_state.getDesiredPosAndAngleTrackbar()
        else:
            [desired_pos, desired_yaw] = current_state.getDesiredPosAndAngle()
        
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

        client.moveToPosition(new_pos[0], new_pos[1], new_pos[2], drone_speed*damping_speed, 0, DrivetrainType.MaxDegreeOfFreedom, YawMode(is_rate=False, yaw_or_rate=degrees(rotation_amount)*damping_yaw_rate), lookahead=-1, adaptive_lookahead=0)

        end = time.time()
        elapsed_time = end - start
        
        if (USE_AIRSIM == True):
            if DELTA_T - elapsed_time > 0:
                time.sleep(DELTA_T - elapsed_time)
            end = time.time()
            elapsed_time = end - start

        #SAVE ALL VALUES OF THIS SIMULATION         
        global mystr
        mystr = str(linecount)+mystr
        f_output.write(mystr)
        
        line = ""
        for i in range(0, bone_pos_GT.shape[1]):
            line = line+'\t'+str(bone_pos_GT[0,i])+'\t'+str(bone_pos_GT[1,i])+'\t'+str(bone_pos_GT[2,i])
        line = line+'\n'
        f_bones.write(line)

        linecount = linecount + 1
        print('linecount', linecount)

        if (USE_AIRSIM == False):
            end_test = client.end

    print('End it!')

if __name__ == "__main__":
    main()

