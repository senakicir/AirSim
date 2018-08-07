from helpers import * 
from State import *
from NonAirSimClient import *
from pose3d_optimizer import *
from pose3d_optimizer_scipy import *
from project_bones import *
import numpy as np
import torch
import cv2 as cv
from torch.autograd import Variable
import time
from scipy.optimize import minimize
from levenberg import *


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
    elif (mode_3d == 3):
        positions, unreal_positions, cov, f_output_str = determine_3d_positions_energy_scipy(mode_2d, measurement_cov_, client, plot_loc, photo_loc)

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

def determine_3d_positions_energy_scipy(mode_2d, measurement_cov_, client, plot_loc = 0, photo_loc = 0):
    unreal_positions, bone_pos_3d_GT, drone_pos_vec, angle = client.getSynchronizedData()
    bone_connections, joint_names, num_of_joints, bone_pos_3d_GT = model_settings(client.model, bone_pos_3d_GT)
    #DONT FORGET THESE CHANGES
    R_drone = euler_to_rotation_matrix(unreal_positions[DRONE_ORIENTATION_IND, 0], unreal_positions[DRONE_ORIENTATION_IND, 1], unreal_positions[DRONE_ORIENTATION_IND, 2])
    C_drone = unreal_positions[DRONE_POS_IND, :]
    C_drone = C_drone[:, np.newaxis]

    #find 2d pose (using openpose or gt)
    bone_2d, heatmap_2d = determine_2d_positions(mode_2d, True, unreal_positions, bone_pos_3d_GT, photo_loc, -1)

    #find 3d pose using liftnet
    pose = torch.cat((torch.t(bone_2d), torch.ones(num_of_joints,1)), 1)
    input_image = cv.imread(photo_loc)
    bone_2d = bone_2d.data.numpy()

    pose3d_lift, cropped_img, cropped_heatmap = liftnet_module.run(input_image, heatmap_2d.cpu().numpy(), pose)
    pose3d_lift = pose3d_lift.view(num_of_joints+2,  -1).permute(1, 0)
    pose3d_lift = rearrange_bones_to_mpi(pose3d_lift, True)
    pose3d_lift = camera_to_world(R_drone, C_drone, pose3d_lift.cpu().data.numpy(), is_torch = False)

    #normalize liftnet pose and save it
    pose3d_lift, _ = normalize_pose(pose3d_lift, joint_names, is_torch = False)
    #hip_pos_lift = pose3d_lift[:, joint_names.index('spine1')]
    #pose3d_lift = pose3d_lift - hip_pos_lift[:, np.newaxis]
    #max_z_lift = np.max(pose3d_lift[2,:])
    #min_z_lift = np.min(pose3d_lift[2,:])
    #pose3d_lift = (pose3d_lift-min_z_lift)/(max_z_lift - min_z_lift)

    if (client.linecount != 0):
        pose3d_ = client.poseList_3d[-1]
    else:
        pose3d_ = take_bone_backprojection(bone_2d, R_drone, C_drone, joint_names)

    if (client.isCalibratingEnergy): 
        client.addNewCalibrationFrame(bone_2d, R_drone, C_drone, pose3d_)
    client.addNewFrame(bone_2d, R_drone, C_drone, pose3d_, pose3d_lift)

    pltpts = {}
    final_loss = np.zeros([1,1])
    if (client.linecount >1):
        #calibration mode parameters
        if (client.isCalibratingEnergy): 
            if (client.linecount < 8):
                num_iterations = 500000
            else:
                num_iterations = 50000
            loss_dict = CALIBRATION_LOSSES
            data_list = client.requiredEstimationData_calibration
            energy_weights = {"proj":0.5, "sym":0.5}

            objective = pose3d_calibration_scipy(client.model, data_list, energy_weights, loss_dict, num_iterations)
            pose3d_init = pose3d_.copy()

         #flight mode parameters
        else:
            num_iterations = 5000#client.iter

            loss_dict = LOSSES
            data_list = client.requiredEstimationData
            lift_list = client.liftPoseList
            energy_weights = client.weights

            objective = pose3d_flight_scipy(client.model, data_list, lift_list, energy_weights, loss_dict, num_iterations, client.WINDOW_SIZE, client.boneLengths)
            
            pose3d_init = np.zeros([client.WINDOW_SIZE, 3, num_of_joints])
            for queue_index, pose3d_ in enumerate(client.poseList_3d):
                pose3d_init[queue_index, :] = pose3d_.copy()

        optimized_res = minimize(objective.forward, pose3d_init, method='Newton-CG', jac={"2-point"}, tol=1e-3, options={'maxiter': num_iterations})
        P_world_scrambled = optimized_res.x
        
        if (client.isCalibratingEnergy):
            P_world = np.reshape(a = P_world_scrambled, newshape = [3, num_of_joints], order = "C")
            client.update3dPos(P_world, all = True)
            if client.linecount > 3:
                for i, bone in enumerate(bone_connections):
                    client.boneLengths[i] = np.sum(np.square(P_world[:, bone[0]] - P_world[:, bone[1]]))
                update_torso_size(0.86*(np.sqrt(np.sum(np.square(P_world[:, joint_names.index('neck')]- P_world[:, joint_names.index('spine1')])))))   
        else:
            P_world_temp = np.reshape(a = P_world_scrambled, newshape = [client.WINDOW_SIZE, 3, num_of_joints], order = "C") 
            P_world = P_world_temp[0,:,:]
            client.update3dPos(P_world)

    #if the frame is the first frame, the energy is found through backprojection
    else:
        P_world = pose3d_
        loss_dict = CALIBRATION_LOSSES
    
    client.error_2d.append(final_loss[0])
    #check,  _ = take_bone_projection_pytorch(P_world, R_drone, C_drone)

    error_3d = np.mean(np.linalg.norm(bone_pos_3d_GT - P_world, axis=0))
    client.error_3d.append(error_3d)
    if (plot_loc != 0):
        #superimpose_on_image(bone_2d.data.numpy(), plot_loc, client.linecount, bone_connections, photo_loc, custom_name="projected_res_", scale = -1, projection=check.data.numpy())
        plot_drone_and_human(bone_pos_3d_GT, P_world, plot_loc, client.linecount, bone_connections, error_3d)
        if (client.linecount >1):
            plot_optimization_losses(objective.pltpts, plot_loc, client.linecount, loss_dict)

    positions = form_positions_dict(angle, drone_pos_vec, P_world[:,0])
    cov = transform_cov_matrix(R_drone, measurement_cov_)
    f_output_str = '\t'+str(unreal_positions[HUMAN_POS_IND, 0]) +'\t'+str(unreal_positions[HUMAN_POS_IND, 1])+'\t'+str(unreal_positions[HUMAN_POS_IND, 2])+'\t'+str(angle[0])+'\t'+str(angle[1])+'\t'+str(angle[2])+'\t'+str(drone_pos_vec.x_val)+'\t'+str(drone_pos_vec.y_val)+'\t'+str(drone_pos_vec.z_val)

    return positions, unreal_positions, cov, f_output_str


def determine_3d_positions_energy(mode_2d, measurement_cov_, client, plot_loc = 0, photo_loc = 0):
    unreal_positions, bone_pos_3d_GT, drone_pos_vec, angle = client.getSynchronizedData()
    bone_connections, joint_names, num_of_joints, bone_pos_3d_GT = model_settings(client.model, bone_pos_3d_GT)
    #DONT FORGET THESE CHANGES
    R_drone = Variable(euler_to_rotation_matrix(unreal_positions[DRONE_ORIENTATION_IND, 0], unreal_positions[DRONE_ORIENTATION_IND, 1], unreal_positions[DRONE_ORIENTATION_IND, 2], returnTensor=True), requires_grad = False)
    C_drone = Variable(torch.FloatTensor([[unreal_positions[DRONE_POS_IND, 0]],[unreal_positions[DRONE_POS_IND, 1]],[unreal_positions[DRONE_POS_IND, 2]]]), requires_grad = False)

    #find 2d pose (using openpose or gt)
    bone_2d, heatmap_2d = determine_2d_positions(mode_2d, True, unreal_positions, bone_pos_3d_GT, photo_loc, -1)

    #find 3d pose using liftnet
    pose = torch.cat((torch.t(bone_2d), torch.ones(num_of_joints,1)), 1)
    input_image = cv.imread(photo_loc)
    pose3d_lift, cropped_img, cropped_heatmap = liftnet_module.run(input_image, heatmap_2d.cpu().numpy(), pose)
    pose3d_lift = pose3d_lift.view(num_of_joints+2,  -1).permute(1, 0)
    pose3d_lift = rearrange_bones_to_mpi(pose3d_lift, True)
    pose3d_lift = camera_to_world(R_drone, C_drone, pose3d_lift.cpu(), is_torch = True)

    #normalize liftnet pose and save it
    #hip_pos_lift = pose3d_lift[:, joint_names.index('spine1')].unsqueeze(1)
    #pose3d_lift = pose3d_lift - hip_pos_lift
    #max_z_lift = torch.max(pose3d_lift[2,:])
    #min_z_lift = torch.min(pose3d_lift[2,:])
    #pose3d_lift = pose3d_lift/(max_z_lift - min_z_lift) #TEST WITHOUT SUB
    pose3d_lift, _ = normalize_pose(pose3d_lift, joint_names, is_torch = True)

    if (client.linecount != 0):
        pose3d_ = client.poseList_3d[-1]
    else:
        pose3d_ = take_bone_backprojection_pytorch(bone_2d, R_drone, C_drone, joint_names)

    if (client.isCalibratingEnergy): 
        client.addNewCalibrationFrame(bone_2d, R_drone, C_drone, pose3d_)
    client.addNewFrame(bone_2d, R_drone, C_drone, pose3d_, pose3d_lift)

    pltpts = {}
    final_loss = np.zeros([1,1])
    if (client.linecount >1):
        #calibration mode parameters
        if (client.isCalibratingEnergy): 
            objective = pose3d_calibration(client.model)
            optimizer = torch.optim.SGD(objective.parameters(), lr = 0.0005, momentum=0.9)
            #optimizer = levenberg(objective.parameters(), lr = 0.0005, momentum=0.9)
            #optimizer = torch.optim.LBFGS(objective.parameters(), lr = 0.0005, max_iter=50)
            if (client.linecount < 8):
                num_iterations = 5000
            else:
                num_iterations = 500
            objective.init_pose3d(pose3d_) 
            loss_dict = CALIBRATION_LOSSES
            data_list = client.requiredEstimationData_calibration
            energy_weights = {"proj":0.5, "sym":0.5}
         #flight mode parameters
        else:
            objective = pose3d_flight(client.boneLengths, client.WINDOW_SIZE, client.model)
            optimizer = torch.optim.SGD(objective.parameters(), lr =client.lr, momentum=client.mu)
            #optimizer = levenberg(objective.parameters(), lr =client.lr, momentum=client.mu)
            #optimizer = torch.optim.LBFGS(objective.parameters(), lr = 0.0005,  max_iter=50)
            num_iterations = client.iter
            #init all 3d pose 
            for queue_index, pose3d_ in enumerate(client.poseList_3d):
                objective.init_pose3d(pose3d_, queue_index)
            loss_dict = LOSSES
            data_list = client.requiredEstimationData
            energy_weights = client.weights

        for loss_key in loss_dict:
            pltpts[loss_key] = np.zeros([num_iterations])

        for i in range(num_iterations):
            def closure():
                outputs = {}
                output = {}
                for loss_key in loss_dict:
                    outputs[loss_key] = []
                    output[loss_key] = 0
                optimizer.zero_grad()
                objective.zero_grad()

                queue_index = 0
                for bone_2d_, R_drone_, C_drone_ in data_list:
                    if (client.isCalibratingEnergy):
                        loss = objective.forward(bone_2d_, R_drone_, C_drone_)
                    else: 
                        pose3d_lift = client.liftPoseList[queue_index]
                        loss = objective.forward(bone_2d_, R_drone_, C_drone_, pose3d_lift, queue_index)

                        if (i == num_iterations - 1 and queue_index == 0):
                            #temp_res_ = objective.pose3d[0, :, :]
                            #temp_hip_pos = temp_res_[:, joint_names.index('spine1')].unsqueeze(1)
                            #temp_res = temp_res_ - temp_hip_pos
                            #max_z = torch.max(temp_res[2,:])
                            #min_z = torch.min(temp_res[2,:])
                            #normalized_pose_3d = temp_res/(max_z - min_z)
                            normalized_pose_3d, _ = normalize_pose(objective.pose3d[0, :, :], joint_names, is_torch = True)
                            plot_drone_and_human(normalized_pose_3d.data.numpy(), pose3d_lift.cpu().numpy(), plot_loc, client.linecount, bone_connections, custom_name="lift_res_", orientation = "z_up")
                    for loss_key in loss_dict:
                        outputs[loss_key].append(loss[loss_key])
                    queue_index += 1

                overall_output = Variable(torch.FloatTensor([0]))
                for loss_key in loss_dict:
                    output[loss_key] = (sum(outputs[loss_key])/len(outputs[loss_key]))
                    overall_output += energy_weights[loss_key]*output[loss_key]
                    pltpts[loss_key][i] = output[loss_key].data.numpy() 
                    if (i == num_iterations - 1):
                        final_loss[0] += energy_weights[loss_key]*np.copy(output[loss_key].data.numpy())

                overall_output.backward(retain_graph = True)
                return overall_output
            optimizer.step(closure)

        if (client.isCalibratingEnergy):
            P_world = objective.pose3d
            client.update3dPos(P_world, all = True)
            if client.linecount > 3:
                for i, bone in enumerate(bone_connections):
                    client.boneLengths[i] = torch.sum(torch.pow(P_world[:, bone[0]] - P_world[:, bone[1]],2)).data 
                update_torso_size(0.86*(torch.sqrt(torch.sum(torch.pow(P_world[:, joint_names.index('neck')].data- P_world[:, joint_names.index('spine1')].data, 2)))))   
        else:
            P_world = objective.pose3d[0, :, :]
            client.update3dPos(P_world)

    #if the frame is the first frame, the energy is found through backprojection
    else:
        P_world = pose3d_
        loss_dict = CALIBRATION_LOSSES
    
    client.error_2d.append(final_loss[0])
    #check,  _ = take_bone_projection_pytorch(P_world, R_drone, C_drone)

    P_world = P_world.cpu().data.numpy()
    error_3d = np.mean(np.linalg.norm(bone_pos_3d_GT - P_world, axis=0))
    client.error_3d.append(error_3d)
    if (plot_loc != 0):
        #superimpose_on_image(bone_2d.data.numpy(), plot_loc, client.linecount, bone_connections, photo_loc, custom_name="projected_res_", scale = -1, projection=check.data.numpy())
        plot_drone_and_human(bone_pos_3d_GT, P_world, plot_loc, client.linecount, bone_connections, error_3d)
        if (client.linecount >1):
            plot_optimization_losses(pltpts, plot_loc, client.linecount, loss_dict)

    positions = form_positions_dict(angle, drone_pos_vec, P_world[:,0])
    cov = transform_cov_matrix(R_drone.cpu().data.numpy(), measurement_cov_)
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
        superimpose_on_image(bone_2d.cpu().numpy(), plot_loc, client.linecount, bone_connections, photo_loc, custom_name="openpose_", scale = scale_)
        pose = torch.cat((torch.t(bone_2d), torch.ones(num_of_joints,1)), 1)

        input_image = cv.imread(photo_loc)

        pose3d_lift, cropped_img, cropped_heatmap = liftnet_module.run(input_image, heatmap_2d.cpu().numpy(), pose)
        #save_heatmaps(cropped_heatmap, client.linecount, plot_loc, custom_name="cropped_heatmap_")
        #save_image(cropped_img, client.linecount, plot_loc, custom_name="cropped_image_")
        
        pose3d_lift = pose3d_lift.view(num_of_joints+2,  -1).permute(1, 0)
        pose3d_lift = rearrange_bones_to_mpi(pose3d_lift, is_torch = True)

        R_drone = Variable(euler_to_rotation_matrix(unreal_positions[DRONE_ORIENTATION_IND, 0], unreal_positions[DRONE_ORIENTATION_IND, 1], unreal_positions[DRONE_ORIENTATION_IND, 2], returnTensor=True), requires_grad = False)
        C_drone = Variable(torch.FloatTensor([[unreal_positions[DRONE_POS_IND, 0]],[unreal_positions[DRONE_POS_IND, 1]],[unreal_positions[DRONE_POS_IND, 2]]]), requires_grad = False)
        pose3d_lift = camera_to_world(R_drone, C_drone, pose3d_lift.cpu(), is_torch = True)

        #hip_pos_lift = pose3d_lift[:, joint_names.index('spine1')].unsqueeze(1)
        #pose3d_lift = pose3d_lift - hip_pos_lift
        #max_z_lift = torch.max(pose3d_lift[2,:])
        #min_z_lift = torch.min(pose3d_lift[2,:])
        #pose3d_lift = (pose3d_lift-min_z_lift)/(max_z_lift - min_z_lift)
        pose3d_lift, _ = normalize_pose(pose3d_lift, joint_names, is_torch = True)
        
        #hip_pos_GT = bone_pos_3d_GT[:, joint_names.index('spine1')]
        #bone_pos_3d_GT = bone_pos_3d_GT - hip_pos_GT[:, np.newaxis]
        #max_z_GT = np.max(bone_pos_3d_GT[2,:])
        #min_z_GT = np.min(bone_pos_3d_GT[2,:])
        #bone_pos_3d_GT = (bone_pos_3d_GT-min_z_GT)/(max_z_GT - min_z_GT)
        bone_pos_3d_GT, _ = normalize_pose(bone_pos_3d_GT, joint_names, is_torch = False)

        plot_drone_and_human(bone_pos_3d_GT, pose3d_lift.cpu().numpy(), plot_loc, client.linecount, bone_connections, custom_name="lift_res_", orientation = "z_up")


    elif (mode_2d == 0):
        superimpose_on_image(bone_2d, plot_loc, client.linecount, bone_connections, photo_loc, custom_name="gt_", scale = scale_)


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