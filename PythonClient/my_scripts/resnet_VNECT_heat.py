import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
import torch.nn.functional as F

import sys
sys.path.insert(0,'../')

import training
import IPython

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNetHeatStream(nn.Module):

    def __init__(self, block, layers, input_key='img_crop', num_maps=17):
        self.input_key = input_key
        
        self.inplanes = 64
        super(ResNetHeatStream, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block,  64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 256, layers[3], stride=1)

        self.toMapsLarge = nn.Sequential(
                        nn.Conv2d(256* block.expansion, 256, kernel_size=1, stride=1, padding=0, bias=True),
                        nn.BatchNorm2d(256),
                        nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=False),
                        nn.BatchNorm2d(128),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0, bias=False),
                        nn.BatchNorm2d(256),
                        nn.ConvTranspose2d(256,128, kernel_size=4, stride=2, padding=1, bias=False), # upsampling
                        nn.BatchNorm2d(128),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
                        nn.BatchNorm2d(128),
                        nn.ReLU(inplace=True),
                        nn.ConvTranspose2d(128,128, kernel_size=4, stride=2, padding=1, bias=False), # upsampling
                        nn.BatchNorm2d(128),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=False),
                        nn.BatchNorm2d(64),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(64, num_maps, kernel_size=3, stride=1, padding=1, bias=False),
                        nn.BatchNorm2d(num_maps),
                        nn.ReLU(inplace=True)
                        )
                 
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x_dict):
        x = x_dict[self.input_key]
        
        x = self.conv1(x) # size /2
        x = self.bn1(x)
        x = self.relu(x) 
        x = self.maxpool(x) # size /2

        x = self.layer1(x) 
        x = self.layer2(x)# size /2
        x = self.layer3(x)# size /2
        
        # heatmap stream
        x_h = self.layer4(x)
        #h = self.toMaps(x_h)
        h = self.toMapsLarge(x_h)
                        
        return {'2d_heat': h}


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
#    model = ResNet(Bottleneck, [3, 4, 6, 1], **kwargs)
    model = ResNetHeatStream(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        weights = torch.load('/cvlabdata1/home/rhodin/code/humanposeannotation/python/pytorch_human_reconstruction/output/dict_class_2D_mpii_2D_singlePersonOnlyFalse_multiPersonAnnotationTrue_cropJitter0o05_lr0o001_augScale0o33RotTrueMirTrueShear0_2018-01-05_12-39/models/network_best_val_t1.pth')
        training.utils.transfer_partial_weights(weights, model)
        #model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetHeatStream(Bottleneck, [3, 4, 23, 1], **kwargs)
    if pretrained:
        training.utils.transfer_partial_weights(model_zoo.load_url(model_urls['resnet101']), model)
        #model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetHeatStream(Bottleneck, [3, 8, 36, 1], **kwargs)
    if pretrained:
        training.utils.transfer_partial_weights(model_zoo.load_url(model_urls['resnet152']), model)
        #model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model
