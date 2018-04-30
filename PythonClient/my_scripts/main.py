import helpers as my_helpers
from human_2dpos import *
from State import *
from NonAirSimClient import *
from pose3d_optimizer import *
from project_bones import *
from determine_positions import *

gt_hv = []
est_hv = []
USE_AIRSIM = False
LENGTH_OF_SIMULATION = 50

def get_client_unreal_values(client, X):
    unreal_positions = np.zeros([5,3])
    if (USE_AIRSIM == True):
        DRONE_INITIAL_POS = client.DRONE_INITIAL_POS
        keys = {b'humanPos': HUMAN_POS_IND, b'dronePos' : DRONE_POS_IND, b'droneOrient': DRONE_ORIENTATION_IND, b'left_arm': L_SHOULDER_IND, b'right_arm': R_SHOULDER_IND}
        for key, value in keys.items():
            element = X[key]
            if (key != b'droneOrient'):
                unreal_positions[value, :] = np.array([element[b'x_val'], element[b'y_val'], -element[b'z_val']])
                unreal_positions[value, :]  = (unreal_positions[value, :] - DRONE_INITIAL_POS)/100
            else:
                unreal_positions[value, :] = np.array([element[b'x_val'], element[b'y_val'], element[b'z_val']])

    else:
        doNothing()
    return unreal_positions

def TakePhoto(client, index, image_folder_loc, saveImage = True):
    if (USE_AIRSIM == True):
        response = client.simGetImages([ImageRequest(0, AirSimImageType.Scene)])
        response = response[0]
        X = response.bones  

        unreal_positions = get_client_unreal_values(client, X)
    
        gt_numbers = [ v for v in X.values() ]
        gt_str = ""
        bone_pos = np.zeros([3, len(gt_numbers)-3])
        DRONE_INITIAL_POS = client.DRONE_INITIAL_POS
        for bone_i in range(0, len(gt_numbers)):
            gt_str = gt_str + str(gt_numbers[bone_i][b'x_val']) + '\t' + str(gt_numbers[bone_i][b'y_val']) + '\t' +  str(gt_numbers[bone_i][b'z_val']) + '\t'
            if (bone_i >= 3):
                bone_pos[:, bone_i-3] = np.array([gt_numbers[bone_i][b'x_val'], gt_numbers[bone_i][b'y_val'], -gt_numbers[bone_i][b'z_val']]) - DRONE_INITIAL_POS

        bone_pos = bone_pos / 100

        drone_orient = client.getPitchRollYaw()    
        drone_pos = client.getPosition()
        client.updateSynchronizedData(unreal_positions, bone_pos, drone_pos, drone_orient)

        if (saveImage == True):
            loc = image_folder_loc + '/img_' + str(index) + '.png'
            AirSimClient.write_file(os.path.normpath(loc), response.image_data_uint8)
    else:
        response = client.simGetImages()
        bone_pos = response.bone_pos
        unreal_positions = response.unreal_positions
        drone_orient = client.getPitchRollYaw()
        drone_pos = client.getPosition()
        client.updateSynchronizedData(unreal_positions, bone_pos, drone_pos, drone_orient)
        gt_str = ""

    return response.image_data_uint8, gt_str

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
        parameters = {"USE_TRACKBAR": False, "USE_GROUNDTRUTH": 1, "USE_AIRSIM": True, "ANIMATION_NUM": 1, "TEST_SET_NAME": "test_set_1", "FILE_NAMES": "", "FOLDER_NAMES": ""}
    
    USE_TRACKBAR = parameters["USE_TRACKBAR"]
    USE_GROUNDTRUTH = parameters["USE_GROUNDTRUTH"] #0 is groundtruth, 1 is mild-GT, 2 is real system
    global USE_AIRSIM
    USE_AIRSIM = parameters["USE_AIRSIM"]
    ANIMATION_NUM = parameters["ANIMATION_NUM"]
    test_set_name = parameters["TEST_SET_NAME"]
    file_names = parameters["FILE_NAMES"]
    folder_names = parameters["FOLDER_NAMES"]

    #connect to the AirSim simulator
    if (USE_AIRSIM == True):
        client = MultirotorClient()
        client.confirmConnection()
        client.enableApiControl(True)
        client.armDisarm(True)
        print('Taking off')
        client.initInitialDronePos()
        client.takeoff()
        client.moveToZ(-z_pos, 2, max_wait_seconds = 5, yaw_mode = YawMode(), lookahead = -1, adaptive_lookahead = 1)
        time.sleep(5)
        client.changeAnimation(ANIMATION_NUM)
    else:
        print(test_set_name)
        filename_bones = 'temp_main/'+test_set_name+'/groundtruth.txt'
        filename_output = 'temp_main/'+test_set_name+'/a_flight.txt'
        client = NonAirSimClient(filename_bones, filename_output)

    filenames_anim = file_names[ANIMATION_NUM]
    foldernames_anim = folder_names[ANIMATION_NUM]
    f_output = open(filenames_anim["f_output"], 'w')
    f_groundtruth = open(filenames_anim["f_groundtruth"], 'w')

    #save drone initial position
    f_groundtruth_prefix = "-1\t" + str(client.DRONE_INITIAL_POS[0,]) + "\t" + str(client.DRONE_INITIAL_POS[1,]) + "\t" + str(client.DRONE_INITIAL_POS[2,])
    for i in range(0,70):
        f_groundtruth_prefix = f_groundtruth_prefix + "\t"
    f_groundtruth.write(f_groundtruth_prefix + "\n")
    photo, _ = TakePhoto(client, 0, foldernames_anim["images"], saveImage=False)

    if (USE_GROUNDTRUTH == 3):
        objective = pose3d_optimizer()
        optimizer = torch.optim.SGD(objective.parameters(), lr = 1000, momentum=0.5)
    else:
        optimizer = 0
        objective = 0

    init_pose3d = True
    initial_positions, _, _, _, _ = determineAllPositions(USE_GROUNDTRUTH, client, MEASUREMENT_NOISE_COV, optimizer, objective, init_pose3d)

    current_state = State(initial_positions, kalman_arguments['KALMAN_PROCESS_NOISE_AMOUNT'])
    
    #shoulder_vector = initial_positions[R_SHOULDER_IND, :] - initial_positions[L_SHOULDER_IND, :] #find initial human orientation!
    #INITIAL_HUMAN_ORIENTATION = np.arctan2(-shoulder_vector[0], shoulder_vector[1]) #in unreal coordinates

    print ('Drone started %.2f m. from the hiker.\n' % current_state.radius)

    #define some variables
    linecount = 0
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

        cv2.createTrackbar('Z','Drone Control', 3, 20, my_helpers.doNothing)
        cv2.setTrackbarPos('Z', 'Drone Control', z_pos)

    while (end_test == False):

        start = time.time()
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

        photo, f_groundtruth_str = TakePhoto(client, linecount, foldernames_anim["images"])

        plot_loc_ = foldernames_anim["superimposed_images"] + '/superimposed_img_' + str(linecount) + '.png'
        if (USE_AIRSIM==True):
            photo_loc_ = foldernames_anim["images"] + '/img_' + str(linecount) + '.png'
        else:
            photo_loc_ = 'temp_main/'+test_set_name+'/images/img_' + str(linecount) + '.png'

        positions, unreal_positions, cov, inFrame, f_output_str = determineAllPositions(USE_GROUNDTRUTH, client, MEASUREMENT_NOISE_COV, optimizer, objective, plot_loc = plot_loc_, photo_loc = photo_loc_)
        inFrame = True #TO DO
        
        current_state.updateState(positions, inFrame, cov) #updates human pos, human orientation, human vel, drone pos
        
        gt_hp.append(unreal_positions[HUMAN_POS_IND, :])
        est_hp.append(current_state.human_pos)
        errors_pos.append(np.linalg.norm(unreal_positions[HUMAN_POS_IND, :]-current_state.human_pos))
        
        if (linecount > 0):
            gt_hv.append((gt_hp[-1]-gt_hp[-2])/DELTA_T)
            est_hv.append(current_state.human_vel)
            errors_vel.append(np.linalg.norm( (gt_hp[-1]-gt_hp[-2])/DELTA_T - current_state.human_vel))

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
        desired_yaw_deg = yaw_candidates[np.argmin(min_diff)]

        #move drone!
        damping_speed = 1
        client.moveToPosition(new_pos[0], new_pos[1], new_pos[2], drone_speed*damping_speed, 0, DrivetrainType.MaxDegreeOfFreedom, YawMode(is_rate=False, yaw_or_rate=desired_yaw_deg), lookahead=-1, adaptive_lookahead=0)

        end = time.time()
        elapsed_time = end - start
        print("elapsed time: ", elapsed_time)
        
        if (USE_AIRSIM == True):
            if DELTA_T - elapsed_time > 0:
                time.sleep(DELTA_T - elapsed_time)
            end = time.time()
            elapsed_time = end - start

        #SAVE ALL VALUES OF THIS SIMULATION       
        f_output_str = str(linecount)+f_output_str + '\n'
        #f_output.write(f_output_str)
        f_groundtruth_str =  str(linecount) + '\t' +f_groundtruth_str + '\n'
        #f_groundtruth.write(f_groundtruth_str)

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
    estimate_folder_name = foldernames_anim["estimates"]
    my_helpers.plotErrorPlots(gt_hp_arr, est_hp_arr, gt_hv_arr, est_hv_arr, errors, estimate_folder_name)

    print('End it!')
    f_groundtruth.close()
    f_output.close()

    client.reset()
    client.changeAnimation(0) #reset animation

    return errors

if __name__ == "__main__":
    kalman_arguments = {"KALMAN_PROCESS_NOISE_AMOUNT" : 5.17947467923e-10, "KALMAN_MEASUREMENT_NOISE_AMOUNT_XY" : 1.38949549437e-08}
    kalman_arguments["KALMAN_MEASUREMENT_NOISE_AMOUNT_Z"] = 517.947467923 * kalman_arguments["KALMAN_MEASUREMENT_NOISE_AMOUNT_XY"]

    animations = [0,1,2,3]
    animations = [0]

    file_names, folder_names = my_helpers.resetAllFolders(animations)

    test_sets = {"test_set_1":0, "test_set_2":1, "test_set_3":2, "test_set_4":3}
    use_airsim = True
    use_groundtruth = 1
    use_trackbar = False
    parameters = {"USE_TRACKBAR": use_trackbar, "USE_GROUNDTRUTH": use_groundtruth, "USE_AIRSIM": use_airsim, "FILE_NAMES": file_names, "FOLDER_NAMES": folder_names}
    if (use_airsim):
         for animation_num in animations:
            parameters["ANIMATION_NUM"]= animation_num
            parameters["TEST_SET_NAME"]= ""
            errors = main(kalman_arguments, parameters)
            print(errors)
    else:
        for test_set, animation_num in test_sets.items():
            parameters["ANIMATION_NUM"]= animation_num
            parameters["TEST_SET_NAME"]= test_set
            errors = main(kalman_arguments, parameters)
            print(errors)