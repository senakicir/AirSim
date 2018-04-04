from main import *

def grid_search():
    #USE_TRACKBAR, USE_GROUNDTRUTH, USE_AIRSIM, PLOT_EVERYTHING, SAVE_VALUES
    other_arguments = [False, 1, False, False, False]

    process_noise_list = np.logspace(-15, -7, 15)
    xy_measurement_noise_list = np.logspace(-15, -5, 15)
    z_xy_ratio_list = np.logspace(1, 5, 10)
    ave_errors_pos = np.zeros([len(process_noise_list),len(xy_measurement_noise_list),len(z_xy_ratio_list)])
    ave_errors_vel = np.zeros([len(process_noise_list),len(xy_measurement_noise_list),len(z_xy_ratio_list)])
    for i, process_noise in enumerate(process_noise_list):
        for j, xy_measurement_noise in enumerate(xy_measurement_noise_list):
            for k, z_xy_ratio in enumerate(z_xy_ratio_list):
                kalman_params= [process_noise, xy_measurement_noise, xy_measurement_noise*z_xy_ratio]
                ave_errors_pos[i,j,k], _, ave_errors_vel[i,j,k], _ = main(kalman_params, other_arguments)

    overall_errors = 0*ave_errors_pos + ave_errors_vel
    ind = np.unravel_index(np.argmin(overall_errors), overall_errors.shape)
    print(np.amin(overall_errors))
    print(process_noise_list[ind[0]], xy_measurement_noise_list[ind[1]], z_xy_ratio_list[ind[2]])

if __name__ == "__main__":
    grid_search()

