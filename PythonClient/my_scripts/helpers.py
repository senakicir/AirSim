import setup_path 
import airsim

import shutil, skimage.io
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time, os
from math import degrees, radians

energy_mode = {1:True, 0:False}
LOSSES = ["proj", "smooth", "bone", "smoothpose", "silly"]
attributes = ['dronePos', 'droneOrient', 'humanPos', 'hip', 'right_up_leg', 'right_leg', 'right_foot', 'left_up_leg', 'left_leg', 'left_foot', 'spine1', 'neck', 'head', 'head_top','left_arm', 'left_forearm', 'left_hand','right_arm','right_forearm','right_hand', 'right_hand_tip', 'left_hand_tip' ,'right_foot_tip' ,'left_foot_tip']
TEST_SETS = {0: "test_set_1", 1: "test_set_2", 2: "test_set_3", 3: "test_set_4"}

def range_angle(angle, limit=360, is_radians = True):
    if is_radians == True:
        angle = degrees(angle)
    if angle > limit:
        angle = angle - 360
    elif angle < limit-360:
        angle = angle + 360
    if is_radians == True:
        angle = radians(angle)
    return angle

def save_bone_positions_2(index, bones, f_output):
    bones = [ v for v in bones.values() ]
    line = str(index)
    for i in range(0, len(bones)):
        line = line+'\t'+str(bones[i][b'x_val'])+'\t'+str(bones[i][b'y_val'])+'\t'+str(bones[i][b'z_val'])
    line = line+'\n'
    f_output.write(line)

def do_nothing(x):
    pass

def reset_all_folders(animation_list, param = ""):
    if param == "":
        date_time_name = time.strftime("%Y-%m-%d-%H-%M")
        folder_names = ['temp_main', 'temp_main/' + date_time_name]
        main_folder_name = 'temp_main/' + date_time_name

    else:
        folder_names = ['temp_main', 'temp_main/' + param]
        main_folder_name = 'temp_main/' + param

    for a_folder_name in folder_names:
        if not os.path.exists(a_folder_name):
            os.makedirs(a_folder_name)
    
    file_names = {}
    folder_names = {}
    for animation in animation_list:
        sub_folder_name = main_folder_name + "/animation_" + str(animation)
        folder_names[animation] = {"images": sub_folder_name + '/images', "estimates": sub_folder_name + '/estimates', "superimposed_images":  sub_folder_name + '/superimposed_images'}
        for a_folder_name in folder_names[animation].values():
            if not os.path.exists(a_folder_name):
                os.makedirs(a_folder_name)
        file_names[animation] = {"f_output": sub_folder_name +  '/a_flight.txt', "f_groundtruth": sub_folder_name +  '/groundtruth.txt'}

    f_notes_name = main_folder_name + "/notes.txt"
    return file_names, folder_names, f_notes_name

def fill_notes(f_notes_name, parameters, energy_parameters):
    f_notes = open(f_notes_name, 'w')
    notes_str = "General Parameters:\n"
    for key, value in parameters.items():
        if (key !=  "FILE_NAMES" and key != "FOLDER_NAMES"):
            notes_str += str(key) + " : " + str(value)
            notes_str += '\n'

    notes_str += '\nEnergy Parameters:\n'

    for key, value in energy_parameters.items():
        notes_str += str(key) + " : " + str(value)
        notes_str += '\n'
    f_notes.write(notes_str)
    

def plot_error(gt_hp_arr, est_hp_arr, gt_hv_arr, est_hv_arr, errors, folder_name):
    #PLOT STUFF HERE AT THE END OF SIMULATION
    fig1 = plt.figure()
    ax = fig1.add_subplot(111, projection='3d')
    p1=ax.plot(est_hp_arr[:, 0], est_hp_arr[:, 1], est_hp_arr[:, 2], c='r', marker='^')
    p2=ax.plot(gt_hp_arr[:, 0], gt_hp_arr[:, 1], gt_hp_arr[:, 2], c='b', marker='^')
    #plt.legend([p1,p2],["estimated", "GT"])

    plt.title(str(errors["error_ave_pos"]))
    plt.savefig(folder_name + '/est_pos_final' + '.png', bbox_inches='tight', pad_inches=0)
    #plt.close()

    fig2 = plt.figure()
    ax = fig2.add_subplot(111, projection='3d')
    p1 = ax.plot(est_hv_arr[:, 0], est_hv_arr[:, 1], est_hv_arr[:, 2], c='r', marker='^')
    p2 = ax.plot(gt_hv_arr[:, 0], gt_hv_arr[:, 1], gt_hv_arr[:, 2], c='b', marker='^')
    plt.legend,([p1,p2],["estimated", "GT"])
    plt.title(str(errors["error_ave_vel"]))
    plt.savefig(folder_name + '/est_vel_final' + '.png', bbox_inches='tight', pad_inches=0)
    plt.close()
    #################

def plot_loss_2d(client, folder_name):
    fig3 = plt.figure()
    plt.plot(client.error_2d)
    plt.title("2d loss for each frame")
    plt.xlabel("Frames")
    plt.ylabel("Error")
    plt.savefig(folder_name + '/loss_plot_2d' + '.png', bbox_inches='tight', pad_inches=0)
    plt.close()

def plot_loss_3d(client, folder_name):
    fig3 = plt.figure()
    plt.plot(client.error_3d)
    plt.title("3d loss for each frame")
    plt.xlabel("Frames")
    plt.ylabel("Error")
    plt.savefig(folder_name + '/loss_plot_3d' + '.png', bbox_inches='tight', pad_inches=0)
    plt.close()


bones_h36m = [[0, 1], [1, 2], [2, 3], [3, 19],
              [0, 4], [4, 5], [5, 6], [6, 20],
              [0, 7], [7, 8], [8, 9], [9, 10],
              [8, 14], [14, 15], [15, 16], [16, 17],
              [8, 11], [11, 12], [12, 13], [13, 18]] #20 connections
left_arm_connections = [[8, 14], [14, 15], [15, 16], [16, 17]]
right_arm_connections  = [[8, 11], [11, 12], [12, 13], [13, 18]]

joint_indices_h36m=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
joint_names_h36m = ['hip','right_up_leg','right_leg','right_foot','left_up_leg','left_leg', 'left_foot','spine1','neck','head', 'head_top', 'left_arm','left_forearm','left_hand','right_arm','right_forearm','right_hand', 'right_hand_tip', 'left_hand_tip', 'right_foot_tip', 'left_foot_tip']


colormap='gist_rainbow'
def superimpose_on_image(numbers, plot_loc, ind, photo_location):
    superimposed_plot_loc = plot_loc + '/superimposed_' + str(ind) + '.png'

    im = plt.imread(photo_location)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(im)
    #plot part
    cmap = plt.get_cmap(colormap)
    colorindex = [17, 0, 5, 9, 15, 2, 18, 10, 12, 4, 14, 13, 11, 3, 7, 8, 16, 6, 1, 19]
    for i, bone in enumerate(bones_h36m):
        color_ = cmap(colorindex[i]/len(bones_h36m))
        ax.plot( numbers[0, bone], numbers[1,bone], color = 'b', linewidth=1)

    plt.savefig(plot_loc, bbox_inches='tight', pad_inches=0)
    plt.close()

def plot_drone_and_human(bones_GT, backprojected_bones, location, ind,  error = -5, custom_name = None):
    fig = plt.figure()
    gs1 = gridspec.GridSpec(1, 1)
    ax = fig.add_subplot(gs1[0], projection='3d')
    
    if custom_name == None:
        name = '/plot3d_'
    else: 
        name = '/'+custom_name

    # maintain aspect ratio
    X = bones_GT[0,:]
    Y = bones_GT[1,:]
    Z = -bones_GT[2,:]
    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() * 0.8
    mid_x = (X.max()+X.min()) * 0.5
    mid_y = (Y.max()+Y.min()) * 0.5
    mid_z = (Z.max()+Z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    #plot drone
    #plot1 = ax.scatter(drone[0], drone[1], drone[2], c='r', marker='o')
    #plot joints
    for i, bone in enumerate(bones_h36m):
        plot1, = ax.plot(bones_GT[0,bone], bones_GT[1,bone], -bones_GT[2,bone], c='b', marker='^', label="GT")
    for i, bone in enumerate(bones_h36m):
        plot2, = ax.plot(backprojected_bones[0,bone], backprojected_bones[1,bone], -backprojected_bones[2,bone], c='r', marker='^', label="Estimate")
    ax.legend(handles=[plot1, plot2])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    if (error != -5):
        plt.title("error: %.4f" %error)

    plot_3d_pos_loc = location + name + str(ind) + '.png'
    plt.savefig(plot_3d_pos_loc)
    plt.close()


def plot_optimization_losses(pltpts, location, ind, calibration_mode=False):
    if (calibration_mode):
        plt.figure()
        x_axis = np.linspace(1,pltpts.shape[0],pltpts.shape[0])
        plt.plot(x_axis, pltpts)
        plt.xlabel("iter")
        plot_3d_pos_loc = location + '/loss_calib_' + str(ind) + '.png'
        plt.savefig(plot_3d_pos_loc)
        plt.close()
    else:
        plt.figure()
        for loss_ind, loss_key in enumerate(LOSSES):
            x_axis = np.linspace(1,pltpts[loss_key].shape[0],pltpts[loss_key].shape[0])
            plt.subplot(1,len(LOSSES),loss_ind+1)
            plt.semilogy(x_axis, pltpts[loss_key])
            plt.xlabel("iter")
            plt.title(loss_key)
        plot_3d_pos_loc = location + '/loss_flight_' + str(ind) + '.png'
        plt.savefig(plot_3d_pos_loc, bbox_inches='tight', pad_inches=0)
        plt.close()

def vector3r_arr_to_dict(input):
    output = dict()
    for attribute in attributes:
        output[attribute] = getattr(input, attribute)
    return output