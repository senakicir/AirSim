from human_2dpos import *
from helpers import *
from State import *
from NonAirSimClient import *
from mpl_toolkits.mplot3d import Axes3D

datetime_folder_name = ''
gt_hv = []
est_hv = []
f_output_str = ''
USE_AIRSIM = False
NUM_OF_ANIMATIONS = 3

def setUseAirSim(use):
    global USE_AIRSIM
    USE_AIRSIM = use

def determineAllPositions(bone_pred, measurement_cov_, client):
    drone_pos_vec = client.getPosition() #airsim gives us the drone coordinates with initial drone loc. as origin
    angle = client.getPitchRollYaw()

    #TO DO;
    R_drone = EulerToRotationMatrix(angle[1], angle[0], angle[2])
    C_drone = np.array([drone_pos_vec.x_val, drone_pos_vec.y_val, drone_pos_vec.z_val]).T

    P_world, cov = TakeBoneBackProjection(bone_pred, R_drone, C_drone, measurement_cov_, 0, False)

    positions = np.zeros([5, 3])    
    positions[DRONE_POS_IND,:] = np.array([drone_pos_vec.x_val, drone_pos_vec.y_val, drone_pos_vec.z_val])
    positions[DRONE_ORIENTATION_IND,:] = np.array([angle[0], angle[1], angle[2]])
    positions[HUMAN_POS_IND,:] = P_world[:,0]
    positions[HUMAN_POS_IND,2] = positions[HUMAN_POS_IND,2]+0.9

    unreal_positions = client.getAllPositions() #airsim gives us the drone coordinates with initial drone loc. as origin

    return positions, unreal_positions, cov

def determineAllPositions_mildly_GT(bone_pos_GT, measurement_cov_, client):
    drone_pos_vec = client.getPosition() #airsim gives us the drone coordinates with initial drone loc. as origin
    angle = client.getPitchRollYaw()

    R_drone = EulerToRotationMatrix(angle[1], angle[0], angle[2]) #pitch roll yaw
    C_drone = np.array([[drone_pos_vec.x_val, drone_pos_vec.y_val, drone_pos_vec.z_val]]).T

    bone_pred, z_val, inFrame = TakeBoneProjection(bone_pos_GT, R_drone, C_drone)
    P_world, cov = TakeBoneBackProjection(bone_pred, R_drone, C_drone, measurement_cov_, z_val, use_z = False)

    positions = np.zeros([5, 3])
    positions[DRONE_POS_IND,:] = np.array([drone_pos_vec.x_val, drone_pos_vec.y_val, drone_pos_vec.z_val])
    positions[DRONE_ORIENTATION_IND,:] = np.array([angle[0], angle[1], angle[2]])
    positions[HUMAN_POS_IND,:] = P_world[:,0]
    positions[HUMAN_POS_IND,2] = positions[HUMAN_POS_IND,2]+0.9

    unreal_positions = client.getAllPositions() #airsim gives us the drone coordinates with initial drone loc. as origin

    global f_output_str
    f_output_str = ''
    f_output_str = '\t'+str(unreal_positions[HUMAN_POS_IND, 0]) +'\t'+str(unreal_positions[HUMAN_POS_IND, 1])+'\t'+str(unreal_positions[HUMAN_POS_IND, 2])+'\t'+str(angle[0])+'\t'+str(angle[1])+'\t'+str(angle[2])+'\t'+str(drone_pos_vec.x_val)+'\t'+str(drone_pos_vec.y_val)+'\t'+str(drone_pos_vec.z_val)
    f_output_str = f_output_str+'\n'

    return positions, unreal_positions, cov, inFrame

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

    global f_output_str
    f_output_str = ''
    f_output_str = '\t'+str(unreal_positions[HUMAN_POS_IND, 0]) +'\t'+str(unreal_positions[HUMAN_POS_IND, 1])+'\t'+str(unreal_positions[HUMAN_POS_IND, 2])+'\t'+str(angle[0])+'\t'+str(angle[1])+'\t'+str(angle[2])+'\t'+str(drone_pos_vec.x_val)+'\t'+str(drone_pos_vec.y_val)+'\t'+str(drone_pos_vec.z_val)
    f_output_str = f_output_str+'\n'

    cov = 1e-20 * np.eye(3,3)
    return positions, unreal_positions, cov

def TakePhoto(client, index, f_groundtruth = None):
    if (USE_AIRSIM == True):
        response = client.simGetImages([ImageRequest(0, AirSimImageType.Scene)])
        response = response[0]
        X = response.bones            

        if (f_groundtruth != None):
            gt_numbers = [ v for v in X.values() ]
            gt_str = ""
            for bone_i in range(0, len(gt_numbers)):
                gt_str = gt_str + str(gt_numbers[bone_i][b'x_val']) + '\t' + str(gt_numbers[bone_i][b'y_val']) + '\t' +  str(gt_numbers[bone_i][b'z_val']) + '\t'
            gt_str = gt_str + '\n'
            f_groundtruth.write(gt_str)

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

def main(kalman_arguments = None, other_arguments = None):

    errors_pos = []
    errors_vel = []
    end_test = False

    if (kalman_arguments == None):
        KALMAN_PROCESS_NOISE_AMOUNT = 1e-5
        KALMAN_MEASUREMENT_NOISE_AMOUNT_XY = 1e-2 
        KALMAN_MEASUREMENT_NOISE_AMOUNT_Z = 1
        MEASUREMENT_NOISE_COV = np.array([[KALMAN_MEASUREMENT_NOISE_AMOUNT_XY, 0, 0], [0, KALMAN_MEASUREMENT_NOISE_AMOUNT_XY, 0], [0, 0, KALMAN_MEASUREMENT_NOISE_AMOUNT_Z]])
    else:
        KALMAN_PROCESS_NOISE_AMOUNT = kalman_arguments[0]
        KALMAN_MEASUREMENT_NOISE_AMOUNT_XY = kalman_arguments[1]
        KALMAN_MEASUREMENT_NOISE_AMOUNT_Z = kalman_arguments[2]
        MEASUREMENT_NOISE_COV = np.array([[KALMAN_MEASUREMENT_NOISE_AMOUNT_XY, 0, 0], [0, KALMAN_MEASUREMENT_NOISE_AMOUNT_XY, 0], [0, 0, KALMAN_MEASUREMENT_NOISE_AMOUNT_Z]])
    if (other_arguments == None):
        USE_TRACKBAR = False
        USE_GROUNDTRUTH = 1 #0 is groundtruth, 1 is mild-GT, 2 is real system
        global USE_AIRSIM
        USE_AIRSIM = False
        PLOT_EVERYTHING = False
        SAVE_VALUES = False
    else:
        USE_TRACKBAR = other_arguments[0]
        USE_GROUNDTRUTH = other_arguments[1] #0 is groundtruth, 1 is mild-GT, 2 is real system
        USE_AIRSIM = other_arguments[2]
        PLOT_EVERYTHING = other_arguments[3]
        SAVE_VALUES = other_arguments[4]
        ANIMATION_NUM = other_arguments[5]

    #connect to the AirSim simulator
    if (USE_AIRSIM == True):
        client = MultirotorClient()
        client.confirmConnection()
        client.enableApiControl(True)
        client.armDisarm(True)
        print('Taking off')
        client.takeoff()
        client.changeAnimation(ANIMATION_NUM)
    else:
        filename_bones = 'temp_main/test_set_1/bones.txt'
        filename_output = 'temp_main/test_set_1/a_flight.txt'
        client = NonAirSimClient(filename_bones, filename_output)

    if (SAVE_VALUES == True):
        global datetime_folder_name
        datetime_folder_name = resetAllFolders()
        f_output = open('temp_main/' + datetime_folder_name + '/a_flight.txt', 'w')
        f_bones = open('temp_main/' + datetime_folder_name + '/bones.txt', 'w')
        f_groundtruth = open('temp_main/' + datetime_folder_name + '/grountruth.txt', 'w')


    #find initial human and drone positions, and find the distance between them, find initial angle of drone
    if (SAVE_VALUES == True):
        photo, bone_pos_GT = TakePhoto(client, 0, f_groundtruth)
    else:
        photo, bone_pos_GT = TakePhoto(client, 0)
    
    if (USE_GROUNDTRUTH == 0):
        # read unreal coordinate positions
        initial_positions, _, _ = determineAllPositions_all_GT(client)
    elif (USE_GROUNDTRUTH == 1):
        initial_positions, _, _, _ = determineAllPositions_mildly_GT(bone_pos_GT, MEASUREMENT_NOISE_COV, client)
    else:
        net = initNet()
        bone_pred = FindJointPos2D(photo, 0, net, make_plot = False)
        initial_positions, _, _ = determineAllPositions(bone_pred, MEASUREMENT_NOISE_COV, client) #translations are relative to initial drone pos now (drone is at the center of coord. system)

    current_state = State(initial_positions, KALMAN_PROCESS_NOISE_AMOUNT)

    #shoulder_vector = initial_positions[R_SHOULDER_IND, :] - initial_positions[L_SHOULDER_IND, :] #find initial human orientation!
    #INITIAL_HUMAN_ORIENTATION = np.arctan2(-shoulder_vector[0], shoulder_vector[1]) #in unreal coordinates

    print ('Drone started %.2f m. from the hiker.\n' % current_state.radius)

    #define some variables
    drone_loc = np.array([0 ,0, 0])
    linecount = 0
    num_of_photos = 1
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

        if (SAVE_VALUES == True):
            photo, bone_pos_GT = TakePhoto(client, num_of_photos, f_groundtruth)
        else:
            photo, bone_pos_GT = TakePhoto(client, num_of_photos)
        num_of_photos = num_of_photos +1

        if (USE_GROUNDTRUTH == 0):
            # read unreal coordinate positions
            positions, unreal_positions, cov = determineAllPositions_all_GT(client)
            inFrame = True
        elif (USE_GROUNDTRUTH == 1):
            positions, unreal_positions, cov, inFrame = determineAllPositions_mildly_GT(bone_pos_GT, MEASUREMENT_NOISE_COV, client)
        else:
            inFrame = True #TO DO
            bone_pred = FindJointPos2D(photo, num_of_photos, net, make_plot = False)
            positions, unreal_positions, cov = determineAllPositions(bone_pred, MEASUREMENT_NOISE_COV, client) #translations are relative to initial drone pos now (drone is at the center of coord. system)

        current_state.updateState(positions, inFrame, cov) #updates human pos, human orientation, human vel, drone pos
        

        gt_hp.append(unreal_positions[HUMAN_POS_IND, :])
        est_hp.append(current_state.human_pos)
        gt_hp_arr = np.asarray(gt_hp)
        est_hp_arr = np.asarray(est_hp)
        errors_pos.append(np.linalg.norm(unreal_positions[HUMAN_POS_IND, :]-current_state.human_pos))
        if (linecount > 5):
            errors_vel.append(np.linalg.norm( (gt_hp_arr[linecount, :]-gt_hp_arr[linecount-1, :])/DELTA_T - current_state.human_vel))

        #PLOT STUFF HERE AND CALCULATE ERROR#
        if PLOT_EVERYTHING == True:
            fig1 = plt.figure()
            ax = fig1.add_subplot(111, projection='3d')
            ax.plot(gt_hp_arr[:, 0], gt_hp_arr[:, 1], gt_hp_arr[:, 2], c='b', marker='^')
            ax.plot(est_hp_arr[:, 0], est_hp_arr[:, 1], est_hp_arr[:, 2], c='r', marker='^')
            error_ave_pos = np.mean(np.asarray(errors_pos))
            plt.title(str(error_ave_pos))
            plt.savefig('temp_main/' + datetime_folder_name + '/estimates/est_pos' + str(linecount) + '.png', bbox_inches='tight', pad_inches=0)
            plt.close()

            if linecount > 0: 
                gt_hv.append((gt_hp_arr[linecount, :]-gt_hp_arr[linecount-1, :])/DELTA_T)
                est_hv.append(current_state.human_vel)
                gt_hv_arr = np.asarray(gt_hv)
                est_hv_arr = np.asarray(est_hv)

                fig2 = plt.figure()
                ax = fig2.add_subplot(111, projection='3d')
                ax.plot(gt_hv_arr[:, 0], gt_hv_arr[:, 1], gt_hv_arr[:, 2], c='b', marker='^')
                ax.plot(est_hv_arr[:, 0], est_hv_arr[:, 1], est_hv_arr[:, 2], c='r', marker='^')
                if (linecount > 5):
                    error_ave_vel = np.mean(np.asarray(errors_vel))
                    plt.title(str(error_ave_vel))
                plt.savefig('temp_main/' + datetime_folder_name + '/estimates/est_vel' + str(linecount) + '.png', bbox_inches='tight', pad_inches=0)
                plt.close()
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
        current_yaw_deg = degrees(angle[2])
        yaw_candidates = np.array([degrees(desired_yaw), degrees(desired_yaw) - 360, degrees(desired_yaw) +360])
        min_diff = np.array([abs(current_yaw_deg -  yaw_candidates[0]), abs(current_yaw_deg -  yaw_candidates[1]), abs(current_yaw_deg -  yaw_candidates[2])])
        desired_yaw_deg_BACKUP = yaw_candidates[np.argmin(min_diff)]

        desired_yaw_deg = RangeAngle(degrees(desired_yaw), 360, False)

        #move drone!
        damping_speed = 1

        print(current_yaw_deg, desired_yaw_deg)
        #client.moveToPosition(new_pos[0], new_pos[1], new_pos[2], drone_speed*damping_speed, 0, DrivetrainType.MaxDegreeOfFreedom, YawMode(is_rate=False, yaw_or_rate=current_yaw_deg+5), lookahead=-1, adaptive_lookahead=0)
        client.moveToPosition(new_pos[0], new_pos[1], new_pos[2], drone_speed*damping_speed, 0, DrivetrainType.MaxDegreeOfFreedom, YawMode(is_rate=False, yaw_or_rate=desired_yaw_deg), lookahead=-1, adaptive_lookahead=0)

        #client.moveToPosition(new_pos[0], new_pos[1], new_pos[2], drone_speed*damping_speed, 0, DrivetrainType.ForwardOnly, YawMode(is_rate=False, yaw_or_rate=270), lookahead=-1, adaptive_lookahead=0)


        end = time.time()
        elapsed_time = end - start
        
        if (USE_AIRSIM == True):
            if DELTA_T - elapsed_time > 0:
                time.sleep(DELTA_T - elapsed_time)
            end = time.time()
            elapsed_time = end - start

        #SAVE ALL VALUES OF THIS SIMULATION       
        if SAVE_VALUES == True:
            global f_output_str
            f_output_str = str(linecount)+f_output_str
            f_output.write(f_output_str)

            line = ""
            for i in range(0, bone_pos_GT.shape[1]):
                line = line+'\t'+str(bone_pos_GT[0,i])+'\t'+str(bone_pos_GT[1,i])+'\t'+str(bone_pos_GT[2,i])
            line = line+'\n'
            f_bones.write(line)

        linecount = linecount + 1
        print('linecount', linecount)

        if (USE_AIRSIM == False):
            end_test = client.end
        else:
            if linecount == 15:
                end_test = True

    print('End it!')
    error_arr_pos = np.asarray(errors_pos)
    error_ave_pos = np.mean(error_arr_pos)
    error_std_pos = np.std(error_arr_pos)

    error_arr_vel = np.asarray(errors_vel)
    error_ave_vel = np.mean(error_arr_vel)
    error_std_vel = np.std(error_arr_vel)

    f_bones.close()
    f_groundtruth.close()
    f_output.close()

    client.reset()

    return error_ave_pos, error_std_pos, error_ave_vel, error_std_vel

if __name__ == "__main__":
    #kalman_arguments = [3.72759372031e-15, 3.72759372031e-12,  3.72759372031e-12*100]
    #kalman_arguments = [1e-11, 1.93069772888e-11,  1.93069772888e-11*10000]
    #kalman_arguments = [2.68269579528e-12, 2.68269579528e-09, 2.68269579528e-09*599.484250319]
    kalman_arguments = [3.72759372031e-11, 7.19685673001e-08, 7.19685673001e-08*77.4263682681]

    setUseAirSim(use = True)

    for animation_num in range(1, NUM_OF_ANIMATIONS+1):
                            #USE_TRACKBAR, USE_GROUNDTRUTH, USE_AIRSIM, PLOT_EVERYTHING, SAVE_VALUES, ANIM_NUM
        other_arguments = [False,       0,              USE_AIRSIM,       True,            True,      animation_num  ]

        (err1, err2, err3, err4) = main(kalman_arguments, other_arguments)
        print((err1, err2, err3, err4))

