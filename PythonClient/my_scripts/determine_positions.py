import helpers as my_helpers
from human_2dpos import *
from State import *
from NonAirSimClient import *
from pose3d_optimizer import *
from project_bones import *
import numpy as np
import torch
from torch.autograd import Variable
import time


def determine_all_positions(mode, client, measurement_cov_ = 0,  plot_loc = 0, photo_loc = 0, linecount = 0):
    inFrame = True
    if (mode == 0):
        positions, unreal_positions, cov, f_output_str = determine_3d_positions_all_GT(client)
    elif (mode == 1):
        positions, unreal_positions, cov, inFrame, f_output_str = determine_3d_positions_backprojection(measurement_cov_, client, plot_loc, photo_loc, linecount)
    elif (mode == 3):            
        positions, unreal_positions, cov, f_output_str = determine_3d_positions_energy(measurement_cov_, client, plot_loc, photo_loc, linecount)

    return positions, unreal_positions, cov, inFrame, f_output_str

def determine_2d_positions(mode, unreal_positions = 0, bone_pos_3d_GT = 0):
    if (mode == 1):
        bone_2d, z_val = find_2d_pose_gt(unreal_positions, bone_pos_3d_GT)
    elif (mode == 3):            
        bone_2d, z_val = find_2d_pose_gt_pytorch(unreal_positions, bone_pos_3d_GT)
    return bone_2d, z_val

def find_2d_pose_gt_pytorch(unreal_positions, bone_pos_3d_GT):
    R_drone_unreal = Variable(euler_to_rotation_matrix(unreal_positions[DRONE_ORIENTATION_IND, 0], unreal_positions[DRONE_ORIENTATION_IND, 1], unreal_positions[DRONE_ORIENTATION_IND, 2], returnTensor=True), requires_grad = False) #pitch roll yaw
    C_drone_unreal = Variable(torch.FloatTensor([[unreal_positions[DRONE_POS_IND, 0]],[unreal_positions[DRONE_POS_IND, 1]],[unreal_positions[DRONE_POS_IND, 2]]]), requires_grad = False)
    bone_pos_GT = Variable(torch.from_numpy(bone_pos_3d_GT).float(), requires_grad = True)
    bone_2d, z_val = take_bone_projection_pytorch(bone_pos_GT, R_drone_unreal, C_drone_unreal)
    return bone_2d, z_val

def find_2d_pose_openpose(unreal_positions, bone_pos_3d_GT):
    #TO DO BIG TIME!
    return 0, 0

def find_2d_pose_gt(unreal_positions, bone_pos_3d_GT):
    R_drone_unreal = euler_to_rotation_matrix(unreal_positions[DRONE_ORIENTATION_IND, 0], unreal_positions[DRONE_ORIENTATION_IND, 1], unreal_positions[DRONE_ORIENTATION_IND, 2])
    C_drone_unreal = unreal_positions[DRONE_POS_IND, :]
    C_drone_unreal = C_drone_unreal[:, np.newaxis]
    bone_2d, z_val, _ = take_bone_projection(bone_pos_3d_GT, R_drone_unreal, C_drone_unreal)
    return bone_2d, z_val

def determine_3d_positions_energy(measurement_cov_, client, plot_loc = 0, photo_loc = 0, linecount = 0):

    unreal_positions, bone_pos_3d_GT, drone_pos_vec, angle = client.getSynchronizedData()
    bone_2d, _ = determine_2d_positions(3, unreal_positions, bone_pos_3d_GT)
    bone_pos_GT = Variable(torch.from_numpy(bone_pos_3d_GT).float(), requires_grad = True)
    additive_noise = torch.normal(torch.ones([1,21])*5, 0.5)
    #bone_pos_GT_noisy = Variable(torch.from_numpy(bone_pos_3d_GT).float()+additive_noise.float(), requires_grad = True)
    #bone_2d_noisy, _ = take_bone_projection_pytorch(bone_pos_GT_noisy, R_drone_unreal, C_drone_unreal)

    #DONT FORGET THESE CHANGES
    #R_drone = Variable(euler_to_rotation_matrix(angle[1], angle[0], angle[2], returnTensor=True), requires_grad = False) #pitch roll yaw
    #C_drone = torch.FloatTensor([[drone_pos_vec.x_val], [drone_pos_vec.y_val], [drone_pos_vec.z_val]])
    R_drone = Variable(euler_to_rotation_matrix(unreal_positions[DRONE_ORIENTATION_IND, 0], unreal_positions[DRONE_ORIENTATION_IND, 1], unreal_positions[DRONE_ORIENTATION_IND, 2], returnTensor=True), requires_grad = False)
    C_drone = Variable(torch.FloatTensor([[unreal_positions[DRONE_POS_IND, 0]],[unreal_positions[DRONE_POS_IND, 1]],[unreal_positions[DRONE_POS_IND, 2]]]), requires_grad = False)

    objective = pose3d_objective()
    optimizer = torch.optim.SGD(objective.parameters(), lr = 0.01, momentum=0.9)

    pose3d_ = take_bone_backprojection_pytorch(bone_2d, R_drone, C_drone, 0)
    objective.init_pose3d(pose3d_)
    client.addNewFrame(bone_2d, R_drone, C_drone)

    if (linecount > 1):# and linecount <= 15):
        num_iterations = 100
        
        pltpts = np.zeros([num_iterations])
        final_loss = np.zeros([1,1])
        final_loss = np.zeros([1,1])
        for i in range(num_iterations):
            def closure():
                #outputs = Variable(torch.FloatTensor([1,len(required_estimation_data)]))
                outputs = []
                optimizer.zero_grad()
                objective.zero_grad()
                for bone_2d_, R_drone_, C_drone_ in client.required_estimation_data:
                    loss = objective.forward(bone_2d_, R_drone_, C_drone_)
                    outputs.append(loss)

                output = sum(outputs)/len(outputs)
                pltpts[i]= output.data.numpy()
                if (i == num_iterations - 1):
                    final_loss[0] = np.copy(output.data.numpy())
                #print("iter ", i, ":", output)
                output.backward(retain_graph = True)
                return output
            optimizer.step(closure)

        #plt.figure()
        #plt.plot(np.linspace(1,num_iterations,num_iterations), pltpts)
        #plt.xlabel("")
        #plt.show()
        #plt.close()
        client.error_2d.append(final_loss[0])
        P_world = objective.pose3d
        check, _ = take_bone_projection_pytorch(P_world, R_drone, C_drone)
    else:
        P_world = pose3d_
        check, _ = take_bone_projection_pytorch(P_world, R_drone, C_drone)

    P_world = P_world.data.numpy()
    error_3d = np.linalg.norm(bone_pos_3d_GT - P_world)
    client.error_3d.append(error_3d)

    if (plot_loc != 0):
        superimposed_plot_loc = plot_loc + '/superimposed_' + str(linecount) + '.png'
        #superimpose_on_image(check.data.numpy(), superimposed_plot_loc, photo_loc)
        plot_3d_pos_loc = plot_loc + '/plot3d_' + str(linecount) + '.png'
        #plot_drone_and_human(bone_pos_3d_GT, P_world, plot_3d_pos_loc, error_3d)

    positions = form_positions_dict(angle, drone_pos_vec, P_world[:,0])
    cov = transform_cov_matrix(R_drone.data.numpy(), measurement_cov_)
    f_output_str = '\t'+str(unreal_positions[HUMAN_POS_IND, 0]) +'\t'+str(unreal_positions[HUMAN_POS_IND, 1])+'\t'+str(unreal_positions[HUMAN_POS_IND, 2])+'\t'+str(angle[0])+'\t'+str(angle[1])+'\t'+str(angle[2])+'\t'+str(drone_pos_vec.x_val)+'\t'+str(drone_pos_vec.y_val)+'\t'+str(drone_pos_vec.z_val)

    return positions, unreal_positions, cov, f_output_str

def determine_3d_positions_backprojection(measurement_cov_, client, plot_loc = 0, photo_loc = 0, linecount = 0):
    inFrame = False #To do

    unreal_positions, bone_pos_3d_GT, drone_pos_vec, angle = client.getSynchronizedData()
    bone_2d, z_val = determine_2d_positions(1, unreal_positions, bone_pos_3d_GT)

    R_drone = euler_to_rotation_matrix(unreal_positions[DRONE_ORIENTATION_IND, 0], unreal_positions[DRONE_ORIENTATION_IND, 1], unreal_positions[DRONE_ORIENTATION_IND, 2])
    C_drone = unreal_positions[DRONE_POS_IND, :]
    C_drone = C_drone[:, np.newaxis]
    #Uncomment for AirSim Metrics
    #R_drone = euler_to_rotation_matrix(angle[1], angle[0], angle[2])
    #C_drone = np.array([[drone_pos_vec.x_val],[drone_pos_vec.y_val],[drone_pos_vec.z_val]])

    P_world = take_bone_backprojection(bone_2d, R_drone, C_drone, z_val, use_z = False)
    error_3d = np.linalg.norm(bone_pos_3d_GT - P_world)
    client.error_3d.append(error_3d)

    if (plot_loc != 0):
        check, _, _ = take_bone_projection(P_world, R_drone, C_drone)
        superimposed_plot_loc = plot_loc + '/superimposed_' + str(linecount) + '.png'
        superimpose_on_image(check, superimposed_plot_loc, photo_loc)
        plot_3d_pos_loc = plot_loc + '/plot3d_' + str(linecount) + '.png'
        plot_drone_and_human(bone_pos_3d_GT, P_world, plot_3d_pos_loc, error_3d)

    cov = transform_cov_matrix(R_drone, measurement_cov_)

    positions = form_positions_dict(angle, drone_pos_vec, P_world[:,0])
    f_output_str = '\t'+str(unreal_positions[HUMAN_POS_IND, 0]) +'\t'+str(unreal_positions[HUMAN_POS_IND, 1])+'\t'+str(unreal_positions[HUMAN_POS_IND, 2])+'\t'+str(angle[0])+'\t'+str(angle[1])+'\t'+str(angle[2])+'\t'+str(drone_pos_vec.x_val)+'\t'+str(drone_pos_vec.y_val)+'\t'+str(drone_pos_vec.z_val)

    return positions, unreal_positions, cov, inFrame, f_output_str

def determine_3d_positions_all_GT(client):
    unreal_positions, _, drone_pos_vec, angle = client.getSynchronizedData()

    positions = form_positions_dict(angle, drone_pos_vec, unreal_positions[HUMAN_POS_IND,:])
    positions[HUMAN_POS_IND,2] = positions[HUMAN_POS_IND,2]
    positions[R_SHOULDER_IND,:] = unreal_positions[R_SHOULDER_IND,:]
    positions[L_SHOULDER_IND,:] = unreal_positions[L_SHOULDER_IND,:]

    f_output_str = '\t'+str(unreal_positions[HUMAN_POS_IND, 0]) +'\t'+str(unreal_positions[HUMAN_POS_IND, 1])+'\t'+str(unreal_positions[HUMAN_POS_IND, 2])+'\t'+str(angle[0])+'\t'+str(angle[1])+'\t'+str(angle[2])+'\t'+str(drone_pos_vec.x_val)+'\t'+str(drone_pos_vec.y_val)+'\t'+str(drone_pos_vec.z_val)
    f_output_str = f_output_str+'\n'

    cov = 1e-20 * np.eye(3,3)
    return positions, unreal_positions, cov, f_output_str

def form_positions_dict(angle, drone_pos_vec, human_pos):
    positions = np.zeros([5, 3])
    positions[DRONE_POS_IND,:] = np.array([drone_pos_vec.x_val, drone_pos_vec.y_val, drone_pos_vec.z_val])
    positions[DRONE_ORIENTATION_IND,:] = np.array([angle[0], angle[1], angle[2]])
    positions[HUMAN_POS_IND,:] = human_pos
    positions[HUMAN_POS_IND,2] = positions[HUMAN_POS_IND,2]
    return positions