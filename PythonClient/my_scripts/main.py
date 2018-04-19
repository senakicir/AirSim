import helpers as my_helpers
from human_2dpos import *
from State import *
from NonAirSimClient import *
from pose3d_optimizer import *
from project_bones import *
from determine_positions import *

datetime_folder_name = ''
gt_hv = []
est_hv = []
USE_AIRSIM = False
NUM_OF_ANIMATIONS = 1
LENGTH_OF_SIMULATION = 50


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
        loc = 'temp_main/' + datetime_folder_name + '/images/img_' + str(index) + '.png'
        AirSimClient.write_file(os.path.normpath(loc), response.image_data_uint8)

        bone_pos = response.bones

    return response.image_data_uint8, bone_pos

def main(kalman_arguments = None, parameters = None):

    errors_pos = []
    errors_vel = []
    errors = {}
    end_test = False

    if (kalman_arguments == None):
        kalman_arguments = {"KALMAN_PROCESS_NOISE_AMOUNT" : 3.72759372031e-11, "KALMAN_MEASUREMENT_NOISE_AMOUNT_XY" : 7.19685673001e-08}
        kalman_arguments["KALMAN_MEASUREMENT_NOISE_AMOUNT_Z"] = 77.4263682681 * kalman_arguments["KALMAN_MEASUREMENT_NOISE_AMOUNT_XY"]
    
    MEASUREMENT_NOISE_COV = np.array([[kalman_arguments["KALMAN_PROCESS_NOISE_AMOUNT"], 0, 0], [0, kalman_arguments["KALMAN_MEASUREMENT_NOISE_AMOUNT_XY"], 0], [0, 0, kalman_arguments["KALMAN_MEASUREMENT_NOISE_AMOUNT_Z"]]])

    if (parameters == None):
        parameters = {"USE_TRACKBAR": False, "USE_GROUNDTRUTH": 1, "USE_AIRSIM": True, "ANIMATION_NUM": 1}
    
    USE_TRACKBAR = parameters["USE_TRACKBAR"]
    USE_GROUNDTRUTH = parameters["USE_GROUNDTRUTH"] #0 is groundtruth, 1 is mild-GT, 2 is real system
    global USE_AIRSIM
    USE_AIRSIM = parameters["USE_AIRSIM"]
    ANIMATION_NUM = parameters["ANIMATION_NUM"]

    #connect to the AirSim simulator
    if (USE_AIRSIM == True):
        client = MultirotorClient()
        client.confirmConnection()
        client.enableApiControl(True)
        client.armDisarm(True)
        print('Taking off')
        client.takeoff()
        #client.changeAnimation(ANIMATION_NUM)
    else:
        filename_bones = 'temp_main/test_set_1/bones.txt'
        filename_output = 'temp_main/test_set_1/a_flight.txt'
        client = NonAirSimClient(filename_bones, filename_output)


    global datetime_folder_name
    datetime_folder_name = my_helpers.resetAllFolders()
    f_output = open('temp_main/' + datetime_folder_name + '/a_flight.txt', 'w')
    f_bones = open('temp_main/' + datetime_folder_name + '/bones.txt', 'w')
    f_groundtruth = open('temp_main/' + datetime_folder_name + '/grountruth.txt', 'w')

    photo, bone_pos_GT = TakePhoto(client, 0, f_groundtruth)

    if (USE_GROUNDTRUTH == 3):
        objective = pose3d_optimizer()
        optimizer = torch.optim.SGD(objective.parameters(), lr = 10000, momentum=0.5)
    else:
        optimizer = 0
        objective = 0

    if (USE_GROUNDTRUTH == 2):
        net = initNet()
        pose_2d = FindJointPos2D(photo, num_of_photos, net, make_plot = False)
    else:
        pose_2d = bone_pos_GT
    initial_positions, _, _, _, _ = determineAllPositions(USE_GROUNDTRUTH, client, pose_2d, MEASUREMENT_NOISE_COV, optimizer, objective)
    
    current_state = State(initial_positions, kalman_arguments['KALMAN_PROCESS_NOISE_AMOUNT'])
    
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
        cv2.createTrackbar('Angle','Drone Control', 0, 360, my_helpers.doNothing)
        #cv2.setTrackbarPos('Angle', 'Angle Control', int(degrees(some_angle-INITIAL_HUMAN_ORIENTATION)))
        cv2.setTrackbarPos('Angle', 'Drone Control', int(degrees(current_state.some_angle)))

        cv2.createTrackbar('Radius','Drone Control', 3, 10, my_helpers.doNothing)
        cv2.setTrackbarPos('Radius', 'Drone Control', int(current_state.radius))

        cv2.createTrackbar('Z','Drone Control', 3, 10, my_helpers.doNothing)
        cv2.setTrackbarPos('Z', 'Drone Control', z_pos)

    while (end_test == False):

        start = time.time()
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

        photo, bone_pos_GT = TakePhoto(client, num_of_photos, f_groundtruth)
        num_of_photos = num_of_photos +1

        if (USE_GROUNDTRUTH == 2):
            pose_2d = FindJointPos2D(photo, num_of_photos, net, make_plot = False)
        else:
            pose_2d = bone_pos_GT
        positions, unreal_positions, cov, inFrame, f_output_str = determineAllPositions(USE_GROUNDTRUTH, client, pose_2d, MEASUREMENT_NOISE_COV, optimizer, objective)
        inFrame = True #TO DO
        
        current_state.updateState(positions, inFrame, cov) #updates human pos, human orientation, human vel, drone pos
        

        gt_hp.append(unreal_positions[HUMAN_POS_IND, :])
        est_hp.append(current_state.human_pos)
        gt_hp_arr = np.asarray(gt_hp)
        est_hp_arr = np.asarray(est_hp)
        
        errors_pos.append(np.linalg.norm(unreal_positions[HUMAN_POS_IND, :]-current_state.human_pos))
        if (linecount > 0):
            gt_hv.append((gt_hp_arr[linecount, :]-gt_hp_arr[linecount-1, :])/DELTA_T)
            est_hv.append(current_state.human_vel)
            errors_vel.append(np.linalg.norm( (gt_hp_arr[linecount, :]-gt_hp_arr[linecount-1, :])/DELTA_T - current_state.human_vel))

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
        if (True): # new version
            yaw_candidates = np.array([degrees(desired_yaw), degrees(desired_yaw) - 360, degrees(desired_yaw) +360])
            min_diff = np.array([abs(current_yaw_deg -  yaw_candidates[0]), abs(current_yaw_deg -  yaw_candidates[1]), abs(current_yaw_deg -  yaw_candidates[2])])
            desired_yaw_deg = yaw_candidates[np.argmin(min_diff)]
        else:
            current_yaw = radians(current_yaw_deg)
            rotation_amount = desired_yaw - current_yaw
            damping_yaw_rate = 1/pi
            desired_yaw_deg = degrees(my_helpers.RangeAngle(rotation_amount, 180, True))*damping_yaw_rate #in degrees

        #move drone!
        damping_speed = 1

        print(current_yaw_deg, desired_yaw_deg)
        client.moveToPosition(new_pos[0], new_pos[1], new_pos[2], drone_speed*damping_speed, 0, DrivetrainType.MaxDegreeOfFreedom, YawMode(is_rate=False, yaw_or_rate=desired_yaw_deg), lookahead=-1, adaptive_lookahead=0)
        #client.moveToPosition(new_pos[0], new_pos[1], new_pos[2], drone_speed*damping_speed, 0, DrivetrainType.ForwardOnly, YawMode(is_rate=False, yaw_or_rate=270), lookahead=-1, adaptive_lookahead=0)

        end = time.time()
        elapsed_time = end - start
        print("elapsed time: ", elapsed_time)
        
        if (USE_AIRSIM == True):
            if DELTA_T - elapsed_time > 0:
                time.sleep(DELTA_T - elapsed_time)
            end = time.time()
            elapsed_time = end - start

        #SAVE ALL VALUES OF THIS SIMULATION       
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
            if (linecount == LENGTH_OF_SIMULATION):
                end_test = True

    #calculate errors
    error_arr_pos = np.asarray(errors_pos)
    errors["error_ave_pos"] = np.mean(error_arr_pos)
    errors["error_std_pos"] = np.std(error_arr_pos)

    error_arr_vel = np.asarray(errors_vel)
    errors["error_ave_vel"] = np.mean(error_arr_vel)
    errors["error_std_vel"] = np.std(error_arr_vel)

    gt_hp_arr = np.asarray(gt_hp)
    est_hp_arr = np.asarray(est_hp)
    gt_hv_arr = np.asarray(gt_hv)
    est_hv_arr = np.asarray(est_hv)
    my_helpers.plotErrorPlots(gt_hp_arr, est_hp_arr, gt_hv_arr, est_hv_arr, errors, datetime_folder_name)

    print('End it!')
    f_bones.close()
    f_groundtruth.close()
    f_output.close()

    client.reset()
    client.changeAnimation(0) #reset animation

    return errors

if __name__ == "__main__":
    kalman_arguments = {"KALMAN_PROCESS_NOISE_AMOUNT" : 3.72759372031e-11, "KALMAN_MEASUREMENT_NOISE_AMOUNT_XY" : 7.19685673001e-08}
    kalman_arguments["KALMAN_MEASUREMENT_NOISE_AMOUNT_Z"] = 77.4263682681 * kalman_arguments["KALMAN_MEASUREMENT_NOISE_AMOUNT_XY"]

    for animation_num in range(1, NUM_OF_ANIMATIONS+1):
        parameters = {"USE_TRACKBAR": False, "USE_GROUNDTRUTH": 3, "USE_AIRSIM": False, "ANIMATION_NUM": animation_num}
        errors = main(kalman_arguments, parameters)
        print(errors)