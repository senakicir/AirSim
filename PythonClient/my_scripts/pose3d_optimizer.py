import torch
import torch.nn 

class pose3d_optimizer(torch.nn.Module):
    def __init__(self):
        super(pose3d_optimizer, self).__init__()

        self.pose3d = torch.nn.Parameter(torch.FloatTensor([3, 16]))

    def forward(self, pose_2d):
        return self.pose3d