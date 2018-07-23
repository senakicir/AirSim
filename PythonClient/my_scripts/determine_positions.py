from helpers import * 
from State import *
from NonAirSimClient import *
from pose3d_optimizer import *
from project_bones import *
import numpy as np
import torch
import cv2 as cv
from torch.autograd import Variable
import time

import util as demo_util

import openpose as openpose_module
import liftnet as liftnet_module

def determine_all_positions(mode_3d, mode_2d, client, measurement_cov_ = 0,  plot_loc = 0, photo_loc = 0):
    inFrame = True
    if (mode_3d == 0):
        positions, unreal_positions, cov, f_output_str = determine_3d_positions_all_GT(mode_2d, client, plot_loc, photo_loc)
    elif (mode_3d == 1):
        positions, unreal_positions, cov, inFrame, f_output_str = determine_3d_positions_backprojection(mode_2d, measurement_cov_, client, plot_loc, photo_loc)
    elif (mode_3d == 2):            
        positions, unreal_positions, cov, f_output_str = determine_3d_positions_energy(mode_2d, measurement_cov_, client, plot_loc, photo_loc)

    return positions, unreal_positions, cov, inFrame, f_output_str

def determine_2d_positions(mode_2d, is_torch = True, unreal_positions = 0, bone_pos_3d_GT = 0, photo_loc = 0, scale = -1):
    if (mode_2d == 0):
        bone_2d, heatmaps = find_2d_pose_gt(unreal_positions, bone_pos_3d_GT, is_torch)
    elif (mode_2d == 1):            
        bone_2d, heatmaps = find_2d_pose_openpose(photo_loc,  scale)
    return bone_2d, heatmaps

def find_2d_pose_gt(unreal_positions, bone_pos_3d_GT, is_torch = True):
    if (is_torch == True):
        R_drone_unreal = Variable(euler_to_rotation_matrix(unreal_positions[DRONE_ORIENTATION_IND, 0], unreal_positions[DRONE_ORIENTATION_IND, 1], unreal_positions[DRONE_ORIENTATION_IND, 2], returnTensor=True), requires_grad = False) #pitch roll yaw
        C_drone_unreal = Variable(torch.FloatTensor([[unreal_positions[DRONE_POS_IND, 0]],[unreal_positions[DRONE_POS_IND, 1]],[unreal_positions[DRONE_POS_IND, 2]]]), requires_grad = False)
        bone_pos_GT = Variable(torch.from_numpy(bone_pos_3d_GT).float(), requires_grad = True)
        bone_2d, heatmaps = take_bone_projection_pytorch(bone_pos_GT, R_drone_unreal, C_drone_unreal)
    else:
        R_drone_unreal = euler_to_rotation_matrix(unreal_positions[DRONE_ORIENTATION_IND, 0], unreal_positions[DRONE_ORIENTATION_IND, 1], unreal_positions[DRONE_ORIENTATION_IND, 2])
        C_drone_unreal = unreal_positions[DRONE_POS_IND, :]
        C_drone_unreal = C_drone_unreal[:, np.newaxis]
        bone_2d, heatmaps = take_bone_projection(bone_pos_3d_GT, R_drone_unreal, C_drone_unreal)
    heatmaps = 0
    return bone_2d, heatmaps

def find_2d_pose_openpose(photo_loc, scale = -1):
    input_image = demo_util.prepare_image(image_path=photo_loc, height=scale)
    poses, heatmaps, model_type = openpose_module.run_only_model(input_image)
    return poses, heatmaps

def determine_3d_positions_energy(mode_2d, measurement_cov_, client, plot_loc = 0, photo_loc = 0):
    unreal_positions, bone_pos_3d_GT, drone_pos_vec, angle = client.getSynchronizedData()
    bone_connections, joint_names, _, bone_pos_3d_GT = model_settings(client.model, bone_pos_3d_GT)

    bone_2d, heatmap_2d = determine_2d_positions(mode_2d, True, unreal_positions, bone_pos_3d_GT, photo_loc, -1)

    #DONT FORGET THESE CHANGES
    #R_drone = Variable(euler_to_rotation_matrix(angle[1], angle[0], angle[2], returnTensor=True), requires_grad = False) #pitch roll yaw
    #C_drone = torch.FloatTensor([[drone_pos_vec.x_val], [drone_pos_vec.y_val], [drone_pos_vec.z_val]])
    R_drone = Variable(euler_to_rotation_matrix(unreal_positions[DRONE_ORIENTATION_IND, 0], unreal_positions[DRONE_ORIENTATION_IND, 1], unreal_positions[DRONE_ORIENTATION_IND, 2], returnTensor=True), requires_grad = False)
    C_drone = Variable(torch.FloatTensor([[unreal_positions[DRONE_POS_IND, 0]],[unreal_positions[DRONE_POS_IND, 1]],[unreal_positions[DRONE_POS_IND, 2]]]), requires_grad = False)

    #if (client.linecount != 0):
    #    pose3d_ = client.poseList_3d[-1]
    #else:
    pose3d_ = take_bone_backprojection_pytorch(bone_2d, R_drone, C_drone, joint_names)
    client.addNewFrame(bone_2d, R_drone, C_drone, pose3d_)

    pltpts = np.zeros([1,1])
    final_loss = np.zeros([1,1])
    if (client.linecount >1):
        #calibration mode
        if (client.isCalibratingEnergy): 
            objective = pose3d_calibration(client.model)
            optimizer = torch.optim.SGD(objective.parameters(), lr = 0.0005, momentum=0.9)

            num_iterations = 1500
            pltpts = np.zeros([num_iterations])
            objective.init_pose3d(pose3d_)

            for i in range(num_iterations):
                def closure():
                    #outputs = Variable(torch.FloatTensor([1,len(client.requiredEstimationData)]))
                    outputs = []
                    optimizer.zero_grad()
                    objective.zero_grad()

                    for bone_2d_, R_drone_, C_drone_ in client.requiredEstimationData:
                        loss = objective.forward(bone_2d_, R_drone_, C_drone_)
                        outputs.append(loss)

                    output = sum(outputs)/len(outputs)
                    pltpts[i]= output.data.numpy()
                    if (i == num_iterations - 1):
                        final_loss[0] = np.copy(output.data.numpy())
                    output.backward(retain_graph = True)
                    return output
                optimizer.step(closure)

            P_world = objective.pose3d
            client.update3dPos(P_world, all = True)
            if client.linecount > 3:
                for i, bone in enumerate(bone_connections):
                    client.boneLengths[i] = torch.sum(torch.pow(P_world[:, bone[0]] - P_world[:, bone[1]],2)).data 
                update_torso_size(0.86*(torch.sqrt(torch.sum(torch.pow(P_world[:, joint_names.index('neck')].data- P_world[:, joint_names.index('hip')].data, 2)))))
                
        #flight mode   
        else:
            objective = pose3d_flight(client.boneLengths, client.WINDOW_SIZE, client.model)
            optimizer = torch.optim.SGD(objective.parameters(), lr =client.lr, momentum=client.mu)
            num_iterations = client.iter
            pltpts = {}

            for loss_key in LOSSES:
                pltpts[loss_key] = np.zeros([num_iterations])

            #init all 3d pose 
            for queue_index, pose3d_ in enumerate(client.poseList_3d):
                objective.init_pose3d(pose3d_, queue_index)

            for i in range(num_iterations):
                def closure():
                    outputs = {}
                    output = {}
                    for loss_key in LOSSES:
                        outputs[loss_key] = []
                        output[loss_key] = 0
                        
                    optimizer.zero_grad()
                    objective.zero_grad()
                    queue_index = 0
                    for bone_2d_, R_drone_, C_drone_ in client.requiredEstimationData:
                        #pose3d_lift = liftnet_module.forward(heatmap_2d)
                        pose3d_lift =0 
                        loss = objective.forward(bone_2d_, R_drone_, C_drone_, pose3d_lift, queue_index)
                        for loss_key in LOSSES:
                            outputs[loss_key].append(loss[loss_key])
                        queue_index += 1

                    overall_output = Variable(torch.FloatTensor([0]))
                    for loss_key in LOSSES:
                        output[loss_key] = (sum(outputs[loss_key])/len(outputs[loss_key]))
                        overall_output += client.weights[loss_key]*output[loss_key]/len(LOSSES)
                        pltpts[loss_key][i] = output[loss_key].data.numpy() 
                        if (i == num_iterations - 1):
                            final_loss[0] += client.weights[loss_key]*np.copy(output[loss_key].data.numpy())/len(LOSSES)

                    overall_output.backward(retain_graph = True)
                    return overall_output
                optimizer.step(closure)
            P_world = objective.pose3d[0, :, :]
            client.update3dPos(P_world)

    #if first frame, 3d pose is found through backproj.     
    else:
        P_world = pose3d_
    
    client.error_2d.append(final_loss[0])
    check,  _ = take_bone_projection_pytorch(P_world, R_drone, C_drone)

    P_world = P_world.data.numpy()

    error_3d = np.mean(np.linalg.norm(bone_pos_3d_GT - P_world, axis=0))
    client.error_3d.append(error_3d)
    if (plot_loc != 0):
        superimpose_on_image([check.data.numpy(), bone_2d.data.numpy()], plot_loc, client.linecount, bone_connections, photo_loc, custom_name="projected_res_", scale = -1)
        plot_drone_and_human(bone_pos_3d_GT, P_world, plot_loc, client.linecount, bone_connections, error_3d)
        plot_optimization_losses(pltpts, plot_loc, client.linecount, client.isCalibratingEnergy)

    positions = form_positions_dict(angle, drone_pos_vec, P_world[:,0])
    cov = transform_cov_matrix(R_drone.data.numpy(), measurement_cov_)
    f_output_str = '\t'+str(unreal_positions[HUMAN_POS_IND, 0]) +'\t'+str(unreal_positions[HUMAN_POS_IND, 1])+'\t'+str(unreal_positions[HUMAN_POS_IND, 2])+'\t'+str(angle[0])+'\t'+str(angle[1])+'\t'+str(angle[2])+'\t'+str(drone_pos_vec.x_val)+'\t'+str(drone_pos_vec.y_val)+'\t'+str(drone_pos_vec.z_val)

    return positions, unreal_positions, cov, f_output_str

def determine_3d_positions_backprojection(mode_2d, measurement_cov_, client, plot_loc = 0, photo_loc = 0):
    unreal_positions, bone_pos_3d_GT, drone_pos_vec, angle = client.getSynchronizedData()
    bone_connections, joint_names, _, bone_pos_3d_GT = model_settings(client.model, bone_pos_3d_GT)

    bone_2d, _ = determine_2d_positions(mode_2d, False, unreal_positions, bone_pos_3d_GT, photo_loc, -1)

    R_drone = euler_to_rotation_matrix(unreal_positions[DRONE_ORIENTATION_IND, 0], unreal_positions[DRONE_ORIENTATION_IND, 1], unreal_positions[DRONE_ORIENTATION_IND, 2])
    C_drone = unreal_positions[DRONE_POS_IND, :]
    C_drone = C_drone[:, np.newaxis]
    #Uncomment for AirSim Metrics
    #R_drone = euler_to_rotation_matrix(angle[1], angle[0], angle[2])
    #C_drone = np.array([[drone_pos_vec.x_val],[drone_pos_vec.y_val],[drone_pos_vec.z_val]])

    P_world = take_bone_backprojection(bone_2d, R_drone, C_drone, joint_names)
    error_3d = np.linalg.norm(bone_pos_3d_GT - P_world, )
    client.error_3d.append(error_3d)

    if (plot_loc != 0):
        check, _, _ = take_bone_projection(P_world, R_drone, C_drone)
        superimpose_on_image([check], plot_loc, client.linecount, bone_connections, photo_loc)
        plot_drone_and_human(bone_pos_3d_GT, P_world, plot_loc, client.linecount, bone_connections, error_3d)

    cov = transform_cov_matrix(R_drone, measurement_cov_)

    positions = form_positions_dict(angle, drone_pos_vec, P_world[:,0])
    f_output_str = '\t'+str(unreal_positions[HUMAN_POS_IND, 0]) +'\t'+str(unreal_positions[HUMAN_POS_IND, 1])+'\t'+str(unreal_positions[HUMAN_POS_IND, 2])+'\t'+str(angle[0])+'\t'+str(angle[1])+'\t'+str(angle[2])+'\t'+str(drone_pos_vec.x_val)+'\t'+str(drone_pos_vec.y_val)+'\t'+str(drone_pos_vec.z_val)

    return positions, unreal_positions, cov, inFrame, f_output_str

def determine_3d_positions_all_GT(mode_2d, client, plot_loc, photo_loc):
    unreal_positions, bone_pos_3d_GT, drone_pos_vec, angle = client.getSynchronizedData()
    bone_connections, joint_names, num_of_joints, bone_pos_3d_GT = model_settings(client.model, bone_pos_3d_GT)
    scale_ = -1

    bone_2d, heatmap_2d = determine_2d_positions(mode_2d, False, unreal_positions, bone_pos_3d_GT, photo_loc, scale_)
    if (mode_2d == 1):
        superimpose_on_image([bone_2d.cpu().numpy()], plot_loc, client.linecount, bone_connections, photo_loc, custom_name="openpose_", scale = scale_)
        #pose = bone_2d[0]
        pose = torch.cat((torch.t(bone_2d), torch.ones(num_of_joints,1)), 1)

        input_image = cv.imread(photo_loc)

        pose3d_lift, cropped_img, cropped_heatmap = liftnet_module.run(input_image, heatmap_2d.cpu().numpy(), pose)
        save_heatmaps(cropped_heatmap, client.linecount, plot_loc, custom_name="cropped_heatmap_")
        save_image(cropped_img, client.linecount, plot_loc, custom_name="cropped_image_")
        
        pose3d_lift = pose3d_lift.view(num_of_joints+2,  -1).permute(1, 0)
        pose3d_lift = rearrange_bones_to_mpi(pose3d_lift, True)
        #pose3d_lift[[0,1,2],:] = pose3d_lift[[0,2,1],:]

        R_drone = Variable(euler_to_rotation_matrix(unreal_positions[DRONE_ORIENTATION_IND, 0], unreal_positions[DRONE_ORIENTATION_IND, 1], unreal_positions[DRONE_ORIENTATION_IND, 2], returnTensor=True), requires_grad = False)
        C_drone = Variable(torch.FloatTensor([[unreal_positions[DRONE_POS_IND, 0]],[unreal_positions[DRONE_POS_IND, 1]],[unreal_positions[DRONE_POS_IND, 2]]]), requires_grad = False)
        bone_pos_3d_GT = world_to_camera(R_drone, C_drone, bone_pos_3d_GT, is_torch = False)

        hip_pos_lift = pose3d_lift[:, joint_names.index('hip')].unsqueeze(1)
        pose3d_lift = pose3d_lift - hip_pos_lift
        max_z_lift = torch.max(pose3d_lift[1,:])
        min_z_lift = torch.min(pose3d_lift[1,:])
        pose3d_lift = (pose3d_lift-min_z_lift)/(max_z_lift - min_z_lift)
        
        hip_pos_GT = bone_pos_3d_GT[:, joint_names.index('hip')]
        bone_pos_3d_GT = bone_pos_3d_GT - hip_pos_GT[:, np.newaxis]
        max_z_GT = np.max(bone_pos_3d_GT[1,:])
        min_z_GT = np.min(bone_pos_3d_GT[1,:])
        bone_pos_3d_GT = (bone_pos_3d_GT-min_z_GT)/(max_z_GT - min_z_GT)

        plot_drone_and_human(bone_pos_3d_GT, pose3d_lift.cpu().numpy(), plot_loc, client.linecount, bone_connections, custom_name="lift_res_", orientation = "y_up")

    #elif (mode_2d == 0):
    #    superimpose_on_image([bone_2d], plot_loc, client.linecount, bone_connections, photo_loc, custom_name="gt_", scale = scale_)


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

def switch_energy(value):
    pass