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

def determineAllPositions(mode, client, measurement_cov_ = 0, optimizer = 0, objective = 0, init_pose3d = False,  plot_loc = 0, photo_loc = 0):
    inFrame = True
    if (mode == 0):
        positions, unreal_positions, cov, f_output_str = determineAllPositions_all_GT(client)
    elif (mode == 1):
        positions, unreal_positions, cov, inFrame, f_output_str = determineAllPositions_backprojection(measurement_cov_, client, plot_loc, photo_loc)
    elif (mode == 2):
        positions, unreal_positions, cov, f_output_str = determineAllPositions_no_GT(measurement_cov_, client)
    elif (mode == 3):            
        positions, unreal_positions, cov, f_output_str = determineAllPositions_use_energy(measurement_cov_, client, optimizer, objective, init_pose3d, plot_loc, photo_loc)

    return positions, unreal_positions, cov, inFrame, f_output_str

def determineAllPositions_use_energy(measurement_cov_, client, optimizer, objective, init_pose3d = False, plot_loc = 0, photo_loc = 0):
    unreal_positions, bone_pos_3d_GT, drone_pos_vec, angle = client.getSynchronizedData()

    R_drone_unreal = Variable(EulerToRotationMatrix(unreal_positions[DRONE_ORIENTATION_IND, 0], unreal_positions[DRONE_ORIENTATION_IND, 1], unreal_positions[DRONE_ORIENTATION_IND, 2], returnTensor=True), requires_grad = False) #pitch roll yaw
    C_drone_unreal = Variable(torch.FloatTensor([[unreal_positions[DRONE_POS_IND, 0]],[unreal_positions[DRONE_POS_IND, 1]],[unreal_positions[DRONE_POS_IND, 2]]]), requires_grad = False)
    additive_noise = torch.normal(torch.ones([1,21])*5, 0.5)
    bone_pos_GT = Variable(torch.from_numpy(bone_pos_3d_GT).float(), requires_grad = True)
    #bone_pos_GT_noisy = Variable(torch.from_numpy(bone_pos_3d_GT).float()+additive_noise.float(), requires_grad = True)

    bone_2d, _ = TakeBoneProjection_Pytorch(bone_pos_GT, R_drone_unreal, C_drone_unreal)
    #bone_2d_noisy, _ = TakeBoneProjection_Pytorch(bone_pos_GT_noisy, R_drone_unreal, C_drone_unreal)

    R_drone = Variable(EulerToRotationMatrix(angle[1], angle[0], angle[2], returnTensor=True), requires_grad = False) #pitch roll yaw
    C_drone = torch.FloatTensor([[drone_pos_vec.x_val], [drone_pos_vec.y_val], [drone_pos_vec.z_val]])

    if (init_pose3d == True):
        pose3d_ = TakeBoneBackProjection_Pytorch(bone_2d, R_drone, C_drone, 0)
        print("Initializing 3d pose")
        objective.init_pose3d(pose3d_)

    num_iterations = 5
    objective.required_estimation_data.append([bone_2d, R_drone, C_drone])
    for i in range(num_iterations):
        def closure():
            #outputs = Variable(torch.FloatTensor([1,len(required_estimation_data)]))
            outputs = []
            optimizer.zero_grad()
            objective.zero_grad()
            #check8 = time.time()
            frame_num = 0
            print("iteration", i, objective.required_estimation_data)
            for bone_2d_, R_drone_, C_drone_ in objective.required_estimation_data:
                outputs.append(objective.forward(bone_2d_, R_drone_, C_drone_))
                frame_num = frame_num + 1
            #check9 = time.time()
            #print("All forward passes take ", (check9-check8)*100)

            output = sum(outputs)/len(outputs)
            #check6 = time.time()
            output.backward(retain_graph = True)
            #check7 = time.time()
            #print("One backward pass takes ", (check7-check6)*100)
            return output
        optimizer.step(closure)

    cov = TransformCovMatrix(R_drone.data.numpy(), measurement_cov_)

    P_world = objective.pose3d
    P_world = P_world.data.numpy()

    if (plot_loc != 0):
        #print("bonepred", bone_pos_3d_GT)
        #SuperImposeOnImage(bone_pred, plot_loc, photo_loc)
        PlotDroneAndHuman(bone_pos_3d_GT, P_world, plot_loc)

    positions = FormPositionsDict(angle, drone_pos_vec, P_world[:,0])

    f_output_str = '\t'+str(unreal_positions[HUMAN_POS_IND, 0]) +'\t'+str(unreal_positions[HUMAN_POS_IND, 1])+'\t'+str(unreal_positions[HUMAN_POS_IND, 2])+'\t'+str(angle[0])+'\t'+str(angle[1])+'\t'+str(angle[2])+'\t'+str(drone_pos_vec.x_val)+'\t'+str(drone_pos_vec.y_val)+'\t'+str(drone_pos_vec.z_val)

    return positions, unreal_positions, cov, f_output_str

def determineAllPositions_no_GT(measurement_cov_, client):
    bone_pred = 0 #HEYYYY
    unreal_positions, bone_pos_3d_GT, drone_pos_vec, angle = client.getSynchronizedData()

    #TO DO;
    R_drone = EulerToRotationMatrix(angle[1], angle[0], angle[2])
    C_drone = np.array([drone_pos_vec.x_val, drone_pos_vec.y_val, drone_pos_vec.z_val]).T

    P_world = TakeBoneBackProjection(bone_pred, R_drone, C_drone, 0, False)
    cov = TransformCovMatrix(R_drone, measurement_cov_)

    positions = FormPositionsDict(angle, drone_pos_vec, P_world[:,0])

    f_output_str = '\t'+str(unreal_positions[HUMAN_POS_IND, 0]) +'\t'+str(unreal_positions[HUMAN_POS_IND, 1])+'\t'+str(unreal_positions[HUMAN_POS_IND, 2])+'\t'+str(angle[0])+'\t'+str(angle[1])+'\t'+str(angle[2])+'\t'+str(drone_pos_vec.x_val)+'\t'+str(drone_pos_vec.y_val)+'\t'+str(drone_pos_vec.z_val)

    return positions, unreal_positions, cov, f_output_str

def determineAllPositions_backprojection(measurement_cov_, client, plot_loc = 0, photo_loc = 0):
    unreal_positions, bone_pos_3d_GT, drone_pos_vec, angle = client.getSynchronizedData()

    R_drone_unreal = EulerToRotationMatrix(unreal_positions[DRONE_ORIENTATION_IND, 0], unreal_positions[DRONE_ORIENTATION_IND, 1], unreal_positions[DRONE_ORIENTATION_IND, 2])
    C_drone_unreal = unreal_positions[DRONE_POS_IND, :]
    C_drone_unreal = C_drone_unreal[:, np.newaxis]
    bone_pred, z_val, inFrame = TakeBoneProjection(bone_pos_3d_GT, R_drone_unreal, C_drone_unreal)

    R_drone = EulerToRotationMatrix(angle[1], angle[0], angle[2])
    C_drone = np.array([[drone_pos_vec.x_val],[drone_pos_vec.y_val],[drone_pos_vec.z_val]])

    P_world = TakeBoneBackProjection(bone_pred, R_drone, C_drone, z_val, use_z = False)
    
    #if (plot_loc != 0):
        #SuperImposeOnImage(bone_pred, plot_loc, photo_loc)
        #PlotDroneAndHuman(bone_pos_3d_GT, P_world, plot_loc)

    cov = TransformCovMatrix(R_drone, measurement_cov_)

    positions = FormPositionsDict(angle, drone_pos_vec, P_world[:,0])

    f_output_str = '\t'+str(unreal_positions[HUMAN_POS_IND, 0]) +'\t'+str(unreal_positions[HUMAN_POS_IND, 1])+'\t'+str(unreal_positions[HUMAN_POS_IND, 2])+'\t'+str(angle[0])+'\t'+str(angle[1])+'\t'+str(angle[2])+'\t'+str(drone_pos_vec.x_val)+'\t'+str(drone_pos_vec.y_val)+'\t'+str(drone_pos_vec.z_val)

    return positions, unreal_positions, cov, inFrame, f_output_str

def determineAllPositions_all_GT(client):
    unreal_positions, _, drone_pos_vec, angle = client.getSynchronizedData()

    positions = FormPositionsDict(angle, drone_pos_vec, unreal_positions[HUMAN_POS_IND,:])
    positions[HUMAN_POS_IND,2] = positions[HUMAN_POS_IND,2]
    positions[R_SHOULDER_IND,:] = unreal_positions[R_SHOULDER_IND,:]
    positions[L_SHOULDER_IND,:] = unreal_positions[L_SHOULDER_IND,:]

    f_output_str = '\t'+str(unreal_positions[HUMAN_POS_IND, 0]) +'\t'+str(unreal_positions[HUMAN_POS_IND, 1])+'\t'+str(unreal_positions[HUMAN_POS_IND, 2])+'\t'+str(angle[0])+'\t'+str(angle[1])+'\t'+str(angle[2])+'\t'+str(drone_pos_vec.x_val)+'\t'+str(drone_pos_vec.y_val)+'\t'+str(drone_pos_vec.z_val)
    f_output_str = f_output_str+'\n'

    cov = 1e-20 * np.eye(3,3)
    return positions, unreal_positions, cov, f_output_str

def FormPositionsDict(angle, drone_pos_vec, human_pos):
    positions = np.zeros([5, 3])
    positions[DRONE_POS_IND,:] = np.array([drone_pos_vec.x_val, drone_pos_vec.y_val, drone_pos_vec.z_val])
    positions[DRONE_ORIENTATION_IND,:] = np.array([angle[0], angle[1], angle[2]])
    positions[HUMAN_POS_IND,:] = human_pos
    positions[HUMAN_POS_IND,2] = positions[HUMAN_POS_IND,2]
    return positions