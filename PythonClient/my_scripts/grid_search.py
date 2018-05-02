from main import *

def grid_search():
    #USE_TRACKBAR, USE_GROUNDTRUTH, USE_AIRSIM, PLOT_EVERYTHING, SAVE_VALUES
    animations = [0]
    file_names, folder_names = my_helpers.reset_all_folders(animations)
    parameters = {"USE_TRACKBAR": False, "USE_GROUNDTRUTH": 1, "USE_AIRSIM": False, "ANIMATION_NUM": 0, "TEST_SET_NAME":"test_set_1", "FILE_NAMES": file_names, "FOLDER_NAMES": folder_names}

    process_noise_list = np.logspace(-15, -7, 15)
    xy_measurement_noise_list = np.logspace(-15, -5, 15)
    z_xy_ratio_list = np.logspace(1, 5, 15)
    ave_errors_pos = np.zeros([len(process_noise_list),len(xy_measurement_noise_list),len(z_xy_ratio_list)])
    ave_errors_vel = np.zeros([len(process_noise_list),len(xy_measurement_noise_list),len(z_xy_ratio_list)])
    for i, process_noise in enumerate(process_noise_list):
        for j, xy_measurement_noise in enumerate(xy_measurement_noise_list):
            for k, z_xy_ratio in enumerate(z_xy_ratio_list):
                kalman_arguments = {"KALMAN_PROCESS_NOISE_AMOUNT" : process_noise, "KALMAN_MEASUREMENT_NOISE_AMOUNT_XY" : xy_measurement_noise}
                kalman_arguments["KALMAN_MEASUREMENT_NOISE_AMOUNT_Z"] = z_xy_ratio * kalman_arguments["KALMAN_MEASUREMENT_NOISE_AMOUNT_XY"]
                errors = main(kalman_arguments, parameters)
                ave_errors_pos[i,j,k], ave_errors_vel[i,j,k]= errors["error_ave_pos"], errors["error_ave_vel"]

    overall_errors = 0.5*ave_errors_pos + 0.5*ave_errors_vel
    ind = np.unravel_index(np.argmin(overall_errors), overall_errors.shape)
    print(np.amin(overall_errors))
    print(process_noise_list[ind[0]], xy_measurement_noise_list[ind[1]], z_xy_ratio_list[ind[2]])

if __name__ == "__main__":
    grid_search()

