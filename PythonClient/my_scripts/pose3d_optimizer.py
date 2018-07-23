import torch
import torch.nn as nn
from helpers import * 
from project_bones import take_bone_projection_pytorch
from torch.autograd import Variable

def mse_loss(input, target):
    return torch.sum(torch.pow((input - target),2)) / input.data.nelement()

class pose3d_calibration(torch.nn.Module):

    def __init__(self, model):
        super(pose3d_calibration, self).__init__()
        _, _, self.NUM_OF_JOINTS, _ = model_settings(model)


        self.pose3d = torch.nn.Parameter(torch.zeros([3, self.NUM_OF_JOINTS]), requires_grad=True)

    def forward(self, pose_2d, R_drone, C_drone):
        projected_2d, _ = take_bone_projection_pytorch(self.pose3d, R_drone, C_drone)
        output = mse_loss(projected_2d, pose_2d)
        return output
    
    def init_pose3d(self, pose3d_):
        self.pose3d.data[:] = pose3d_.data[:]


class pose3d_flight(torch.nn.Module):

    def __init__(self, bone_lengths_, window_size_, model):
        super(pose3d_flight, self).__init__()
        self.bone_connections, _, self.NUM_OF_JOINTS, _ = model_settings(model)
        self.window_size = window_size_
        self.pose3d = torch.nn.Parameter(torch.zeros([self.window_size, 3, self.NUM_OF_JOINTS]), requires_grad=True)
        self.bone_lengths = Variable(bone_lengths_, requires_grad = False)

    def forward(self, pose_2d, R_drone, C_drone, pose3d_lift, queue_index):
        #projection loss
        outputs = {}
        for loss in LOSSES:
            outputs[loss] = 0
        projected_2d, _ = take_bone_projection_pytorch(self.pose3d[queue_index, :, :], R_drone, C_drone)
        outputs["proj"] = mse_loss(projected_2d, pose_2d)

        #smoothness
        #if (queue_index != self.window_size-1 and queue_index != 0):
        #    outputs["smooth"] = mse_loss(self.pose3d[queue_index, :, :] - self.pose3d[queue_index+1, :, :], self.pose3d[queue_index-1, :, :]- self.pose3d[queue_index, :, :])
        if (queue_index != self.window_size-1 and queue_index != 0):
            outputs["smooth"] = mse_loss(self.pose3d[queue_index, :, :], self.pose3d[queue_index+1, :, :]) +  mse_loss(self.pose3d[queue_index-1, :, :], self.pose3d[queue_index, :, :])
        elif (queue_index != self.window_size-1 ):
            outputs["smooth"] = mse_loss(self.pose3d[queue_index, :, :], self.pose3d[queue_index+1, :, :])
        elif (queue_index != 0):
            outputs["smooth"] = mse_loss(self.pose3d[queue_index-1, :, :], self.pose3d[queue_index, :, :])

        #bone length consistency 
        bonelosses = Variable(torch.zeros([self.NUM_OF_JOINTS-1,1]), requires_grad = False)
        for i, bone in enumerate(self.bone_connections):
            length_of_bone = (torch.sum(torch.pow(self.pose3d[queue_index, :, bone[0]] - self.pose3d[queue_index, :, bone[1]], 2)))
            bonelosses[i] = torch.pow((self.bone_lengths[i] - length_of_bone),2)
        outputs["bone"] = torch.sum(bonelosses)/bonelosses.data.nelement()

        hip = self.pose3d[queue_index, :, 0]
        temp_pose3d_t = torch.sub(self.pose3d[queue_index, :, :], hip.unsqueeze(1))
        if (queue_index != self.window_size-1 and queue_index != 0):
            temp_pose3d_t_p_1 = torch.sub(self.pose3d[queue_index+1, :, :], self.pose3d[queue_index+1, :, 0].unsqueeze(1))
            temp_pose3d_t_m_1 = torch.sub(self.pose3d[queue_index-1, :, :], self.pose3d[queue_index-1, :, 0].unsqueeze(1))
            outputs["smoothpose"] = mse_loss(temp_pose3d_t, temp_pose3d_t_p_1) +  mse_loss(temp_pose3d_t_m_1, temp_pose3d_t)
        elif (queue_index != self.window_size-1 ):
            temp_pose3d_t_p_1 = torch.sub(self.pose3d[queue_index+1, :, :], self.pose3d[queue_index+1, :, 0].unsqueeze(1))
            outputs["smoothpose"] = mse_loss(temp_pose3d_t, temp_pose3d_t_p_1)
        elif (queue_index != 0):
            temp_pose3d_t_m_1 = torch.sub(self.pose3d[queue_index-1, :, :], self.pose3d[queue_index-1, :, 0].unsqueeze(1))
            outputs["smoothpose"] = mse_loss(temp_pose3d_t_m_1, temp_pose3d_t)

        #normalize pose
        #mean_3d = torch.mean(self.pose3d[queue_index, :, :].unsqueeze(0), dim=2).unsqueeze(2)
        #std_3d = torch.std(self.pose3d[queue_index, :, :].unsqueeze(0), dim=2).unsqueeze(2)
        #outputs_norm = (self.pose3d[queue_index, :, :].unsqueeze(0) - mean_3d)/std_3d
        #bone_3d_temp = torch.sub(outputs_norm, hip.unsqueeze(1))

        #outputs["lift"]= mse_loss(pose3d_lift.cpu(), temp_pose3d_t)

        return outputs
    
    def init_pose3d(self, pose3d_, queue_index):
        self.pose3d.data[queue_index, :, :] = pose3d_.data[:]


