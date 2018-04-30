import torch
import torch.nn as nn
from project_bones import TakeBoneProjection_Pytorch
from torch.autograd import Variable

def mse_loss(input, target):
    return torch.sum(torch.pow((input - target),2)) / input.data.nelement()

class pose3d_optimizer(torch.nn.Module):
    def __init__(self):
        super(pose3d_optimizer, self).__init__()
        self.pose3d = torch.nn.Parameter(torch.zeros([3, 21]), requires_grad=True)
        self.loss = nn.MSELoss()
        self.required_estimation_data = []

    def forward(self, pose_2d, R_drone, C_drone):
        projected_3d, _ = TakeBoneProjection_Pytorch(self.pose3d, R_drone, C_drone)
        output = mse_loss(projected_3d, pose_2d)
        return output
    
    def init_pose3d(self, pose3d_):
        self.pose3d = torch.nn.Parameter(pose3d_.data, requires_grad=True)
