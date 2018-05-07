import torch
import torch.nn as nn
import helpers as my_helpers
from project_bones import take_bone_projection_pytorch
from autograd import grad
from torch.autograd import Variable


NUM_OF_BONES = 21
def mse_loss(input, target):
    return torch.sum(torch.pow((input - target),2)) / input.data.nelement()

class pose3d_calibration(torch.nn.Module):
    def __init__(self):
        super(pose3d_calibration, self).__init__()
        self.pose3d = torch.nn.Parameter(torch.zeros([3, NUM_OF_BONES]), requires_grad=True)

    def forward(self, pose_2d, R_drone, C_drone):
        projected_2d, _ = take_bone_projection_pytorch(self.pose3d, R_drone, C_drone)
        output = mse_loss(projected_2d, pose_2d)
        return output
    
    def init_pose3d(self, pose3d_):
        self.pose3d.data[:] = pose3d_.data[:]

#experimental
class pose3d_calibration_new():
    def __init__(self):
        super(pose3d_calibration_new, self).__init__()
        self.pose3d = torch.nn.Parameter(torch.zeros([3, NUM_OF_BONES]), requires_grad=True)

    def backproj_loss(self, pose_2d, R_drone, C_drone):
        projected_2d, _ = take_bone_projection_pytorch(self.pose3d, R_drone, C_drone)
        output = mse_loss(projected_2d, pose_2d)
        return output
    
    def init_pose3d(self, pose3d_):
        self.pose3d.data[:] = pose3d_.data[:]

class pose3d_flight(torch.nn.Module):
    def __init__(self, bone_lengths_):
        super(pose3d_flight, self).__init__()
        self.pose3d = torch.nn.Parameter(torch.zeros([6, 3, NUM_OF_BONES]), requires_grad=True)
        self.bone_lengths = Variable(bone_lengths_, requires_grad = False)

    def forward(self, pose_2d, R_drone, C_drone, queue_index):
        #projection loss
        projected_2d, _ = take_bone_projection_pytorch(self.pose3d[queue_index, :, :], R_drone, C_drone)
        output1 = mse_loss(projected_2d, pose_2d)
        #smoothness
        output2 = 0
        if (queue_index != 5):
            output2 = mse_loss(self.pose3d[queue_index, :, :], self.pose3d[queue_index+1, :, :])
        #bone length consistency 
        bonelosses = Variable(torch.zeros([NUM_OF_BONES-1,1]), requires_grad = False)
        for i, bone in enumerate(my_helpers.bones_h36m):
            length_of_bone = torch.sum(torch.pow(self.pose3d[queue_index, :, bone[0]] - self.pose3d[queue_index, :, bone[1]], 2)) 
            bonelosses[i] = torch.pow((self.bone_lengths[i] - length_of_bone),2)
        output3 = torch.sum(bonelosses)/bonelosses.data.nelement()

        output = output1 + output2 + output3
        return output
    
    def init_pose3d(self, pose3d_, queue_index):
        self.pose3d.data[queue_index, :, :] = pose3d_.data[:]


