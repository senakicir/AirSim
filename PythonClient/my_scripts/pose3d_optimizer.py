import torch
import torch.nn as nn
from project_bones import take_bone_projection_pytorch
from torch.autograd import Variable

def mse_loss(input, target):
    return torch.sum(torch.pow((input - target),2)) / input.data.nelement()

class pose3d_objective(torch.nn.Module):
    def __init__(self):
        super(pose3d_objective, self).__init__()
        self.pose3d = torch.nn.Parameter(torch.zeros([3, 21]), requires_grad=True)
        self.loss = nn.MSELoss()
        self.max_num_of_frames = 10

    def forward(self, pose_2d, R_drone, C_drone):
        projected_3d, _ = take_bone_projection_pytorch(self.pose3d, R_drone, C_drone)
        output = mse_loss(projected_3d, pose_2d)
        return output
    
    def init_pose3d(self, pose3d_):
        self.pose3d.data[:] = pose3d_.data[:]

