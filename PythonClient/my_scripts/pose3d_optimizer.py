import torch
import torch.nn as nn
import helpers as my_helpers
from project_bones import take_bone_projection_pytorch
from torch.autograd import Variable


NUM_OF_JOINTS = 21
def mse_loss(input, target):
    return torch.sum(torch.pow((input - target),2)) / input.data.nelement()

class pose3d_calibration(torch.nn.Module):
    def __init__(self):
        super(pose3d_calibration, self).__init__()
        self.pose3d = torch.nn.Parameter(torch.zeros([3, NUM_OF_JOINTS]), requires_grad=True)

    def forward(self, pose_2d, R_drone, C_drone):
        projected_2d, _ = take_bone_projection_pytorch(self.pose3d, R_drone, C_drone)
        output = mse_loss(projected_2d, pose_2d)
        return output
    
    def init_pose3d(self, pose3d_):
        self.pose3d.data[:] = pose3d_.data[:]


class pose3d_flight(torch.nn.Module):
    def __init__(self, bone_lengths_, WINDOW_SIZE):
        super(pose3d_flight, self).__init__()
        self.pose3d = torch.nn.Parameter(torch.zeros([WINDOW_SIZE, 3, NUM_OF_JOINTS]), requires_grad=True)
        self.bone_lengths = Variable(bone_lengths_, requires_grad = False)

    def forward(self, pose_2d, R_drone, C_drone, queue_index):
        #projection loss
        outputs = {}
        for loss in my_helpers.LOSSES:
            outputs[loss] = 0
        projected_2d, _ = take_bone_projection_pytorch(self.pose3d[queue_index, :, :], R_drone, C_drone)
        outputs["proj"] = mse_loss(projected_2d, pose_2d)
        #smoothness
        if (queue_index != WINDOW_SIZE-1 and queue_index != 0):
            outputs["smooth"] = mse_loss(self.pose3d[queue_index, :, :], self.pose3d[queue_index+1, :, :]) +  mse_loss(self.pose3d[queue_index-1, :, :], self.pose3d[queue_index, :, :])
        elif (queue_index != WINDOW_SIZE-1 ):
            outputs["smooth"] = mse_loss(self.pose3d[queue_index, :, :], self.pose3d[queue_index+1, :, :])
        elif (queue_index != 0):
            outputs["smooth"] = mse_loss(self.pose3d[queue_index-1, :, :], self.pose3d[queue_index, :, :])

        #bone length consistency 
        #bonelosses = Variable(torch.zeros([NUM_OF_JOINTS-1,1]), requires_grad = False)
        #for i, bone in enumerate(my_helpers.bones_h36m):
        #    length_of_bone = (torch.sum(torch.pow(self.pose3d[queue_index, :, bone[0]] - self.pose3d[queue_index, :, bone[1]], 2)))
        #    bonelosses[i] = torch.pow((self.bone_lengths[i] - length_of_bone),2)
        #outputs["bone"] = torch.sum(bonelosses)/bonelosses.data.nelement()

        return outputs
    
    def init_pose3d(self, pose3d_, queue_index):
        self.pose3d.data[queue_index, :, :] = pose3d_.data[:]


