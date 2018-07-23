import torch.nn as nn
from torch.nn import Conv2d
from torch.nn import Linear
from torch.nn import ReLU
from torch.nn import MaxPool2d
from torch.nn import Dropout
from torch.nn import BatchNorm1d
import torch.nn.init as init


class SillyNet(nn.Module):
    def __init__(self, nb_channels=21, num_joints=21, nb_dims=3): 
        super(SillyNet, self).__init__()

        self.lin_input = Linear(2 * 21, 1024)
      
        #linear 1
        self.lin1 = Linear(1024, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.dropout1 = Dropout(inplace=True, p=0.5)

        #linear 2
        self.lin2 = Linear(1024, 1024)
        self.bn2 = nn.BatchNorm1d(1024)
        self.dropout2 = Dropout(inplace=True, p=0.5)

        #linear 3
        self.lin3 = Linear(1024, 1024)
        self.bn3 = nn.BatchNorm1d(1024)
        self.dropout3 = Dropout(inplace=True, p=0.5)

        #linear 4
        self.lin4 = Linear(1024, 1024)
        self.bn4 = nn.BatchNorm1d(1024)
        self.dropout4 = Dropout(inplace=True, p=0.5)
 
        self.linout = Linear(1024, 21 * 3)

    def forward(self, flat_input):
        input_transformed = self.lin_input(flat_input)

        residual = input_transformed

        out = self.lin1(input_transformed)
        out = self.bn1(out)
        out = self.dropout1(out)

        out = self.lin2(out)
        out = self.bn2(out)
        out = self.dropout2(out)

        out = out + residual
        residual2 = out

        out = self.lin3(out)
        out = self.bn3(out)
        out = self.dropout3(out)

        out = self.lin4(out)
        out = self.bn4(out)
        out = self.dropout4(out)

        out = out + residual2

        final_out = self.linout(out) 
        return final_out
