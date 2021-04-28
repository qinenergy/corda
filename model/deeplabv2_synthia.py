"""
This is the implementation of DeepLabv2 without multi-scale inputs. This implementation uses ResNet-101 by default.
"""

import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
import numpy as np
from model.layers import SEBlock, SABlock

affine_par = True


NUM_OUTPUT = {"S": 19, "D": 1, "D_src": 1}
TASKNAMES = ["S", "D"]

def outS(i):
    i = int(i)
    i = (i+1)/2
    i = int(np.ceil((i+1)/2.0))
    i = (i+1)/2
    return i

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, affine = affine_par)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, affine = affine_par)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        print(out.shape)
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

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False) # change
        self.bn1 = nn.BatchNorm2d(planes,affine = affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False

        padding = dilation
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, # change
                               padding=padding, bias=False, dilation = dilation)
        self.bn2 = nn.BatchNorm2d(planes,affine = affine_par)
        for i in self.bn2.parameters():
            i.requires_grad = False
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4, affine = affine_par)
        for i in self.bn3.parameters():
            i.requires_grad = False
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


class BottleneckPad(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(BottleneckPad, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False) # change
        self.bn1 = nn.BatchNorm2d(planes,affine = affine_par)

        padding = dilation
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, # change
                               padding=padding, bias=False, dilation = dilation)
        self.bn2 = nn.BatchNorm2d(planes,affine = affine_par)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4, affine = affine_par)
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

class Classifier_Module(nn.Module):

    def __init__(self, dilation_series, padding_series, num_classes):
        super(Classifier_Module, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(nn.Conv2d(256, num_classes, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias = True))

        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list)-1):
            out += self.conv2d_list[i+1](x)
            return out



class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, affine = affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True) # change
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)
        #self.layer5 = self._make_pred_layer(Classifier_Module, [6,12,18,24],[6,12,18,24],num_classes)
        # PADNET stuff
        self.tasks = TASKNAMES
        self.auxilary_tasks = TASKNAMES
        self.channels = 2048#backbone_channels
        # Task-specific heads for initial prediction
        self.initial_task_prediction_heads = InitialTaskPredictionModule(self.auxilary_tasks, self.channels)
        # Multi-modal distillation
        self.multi_modal_distillation = MultiTaskDistillationModule(self.tasks, self.auxilary_tasks, 256)
        # Task-specific heads for final prediction
        heads = {}
        for task in self.tasks + ["D_src"]:
            heads[task] = self._make_pred_layer(Classifier_Module, [6,12,18,24],[6,12,18,24], NUM_OUTPUT[task]) 
        self.heads = nn.ModuleDict(heads)        


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation == 2 or dilation == 4:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion,affine = affine_par))
        for i in downsample._modules['1'].parameters():
            i.requires_grad = False
        layers = []
        layers.append(block(self.inplanes, planes, stride,dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)
    def _make_pred_layer(self,block, dilation_series, padding_series,num_classes):
        return block(dilation_series,padding_series,num_classes)

    def forward(self, x):
        img_size = x.size()[-2:]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        #x = self.layer5(x)
        out = {}
        # Initial predictions for every task including auxilary tasks
        x = self.initial_task_prediction_heads(x)
        for task in self.auxilary_tasks:
            out['initial_%s' %(task)] = x[task]
        out["initial_D_src"] = x["D_src"]
        # Refine features through multi-modal distillation
        x = self.multi_modal_distillation(x)

        # Make final prediction with task-specific heads
        for task in self.tasks:
            out[task] = self.heads[task](x[task]) 
            if task == "D":
                out["D_src"] = self.heads["D_src"](x[task])

        return out

    def get_1x_lr_params_NOscale(self):
        """
        This generator returns all the parameters of the net except for
        the last classification layer. Note that for each batchnorm layer,
        requires_grad is set to False in deeplab_resnet.py, therefore this function does not return
        any batchnorm parameter
        """
        b = []

        b.append(self.conv1)
        b.append(self.bn1)
        b.append(self.layer1)
        b.append(self.layer2)
        b.append(self.layer3)
        b.append(self.layer4)


        for i in range(len(b)):
            for j in b[i].modules():
                jj = 0
                for k in j.parameters():
                    jj+=1
                    if k.requires_grad:
                        yield k

    def get_10x_lr_params(self):
        """
        This generator returns all the parameters for the last layer of the net,
        which does the classification of pixel into classes
        """
        b = []
        b.append(self.multi_modal_distillation.parameters())
        b.append(self.initial_task_prediction_heads.parameters())
        for task in self.tasks + ["D_src"]:
            b.append(self.heads[task].parameters())

        for j in range(len(b)):
            for i in b[j]:
                yield i



    def optim_parameters(self, args):
        return [{'params': self.get_1x_lr_params_NOscale(), 'lr': args.learning_rate},
                {'params': self.get_10x_lr_params(), 'lr': 10*args.learning_rate}]


def Res_Deeplab(num_classes):
    model = ResNet(Bottleneck,[3, 4, 23, 3], num_classes)
    return model



# The following modules are directly taken from https://github.com/SimonVandenhende/Multi-Task-Learning-PyTorch
class InitialTaskPredictionModule(nn.Module):
    """
        Make the initial task predictions from the backbone features.
    """
    def __init__(self, tasks, input_channels, intermediate_channels=256):
        super(InitialTaskPredictionModule, self).__init__() 
        self.tasks = tasks 
        layers = {}
        conv_out = {}
        
        for task in self.tasks:
            if input_channels != intermediate_channels:
                downsample = nn.Sequential(nn.Conv2d(input_channels, intermediate_channels, kernel_size=1,
                                                    stride=1, bias=False), nn.BatchNorm2d(intermediate_channels))
            else:
                downsample = None
            bottleneck1 = BottleneckPad(input_channels, intermediate_channels//4, downsample=downsample)
            bottleneck2 = BottleneckPad(intermediate_channels, intermediate_channels//4, downsample=None)
            conv_out_ = nn.Conv2d(intermediate_channels, NUM_OUTPUT[task], 1)
            layers[task] = nn.Sequential(bottleneck1, bottleneck2)
            conv_out[task] = conv_out_
        conv_out["D_src"] = nn.Conv2d(intermediate_channels, NUM_OUTPUT["D"], 1)
        self.layers = nn.ModuleDict(layers)
        self.conv_out = nn.ModuleDict(conv_out)


    def forward(self, x):
        out = {}
        
        for task in self.tasks:
            out['features_%s' %(task)] = self.layers[task](x)
            out[task] = self.conv_out[task](out['features_%s' %(task)])
        out["D_src"] = self.conv_out["D_src"](out['features_D'])
        return out 


class MultiTaskDistillationModule(nn.Module):
    """
        Perform Multi-Task Distillation
        We apply an attention mask to features from other tasks and
        add the result as a residual.
    """
    def __init__(self, tasks, auxilary_tasks, channels):
        super(MultiTaskDistillationModule, self).__init__()
        self.tasks = tasks
        self.auxilary_tasks = auxilary_tasks
        self.self_attention = {}
        
        for t in self.tasks:
            other_tasks = [a for a in self.auxilary_tasks if a != t]
            self.self_attention[t] = nn.ModuleDict({a: SABlock(channels, channels) for a in other_tasks})
        self.self_attention = nn.ModuleDict(self.self_attention)


    def forward(self, x):
        adapters = {t: {a: self.self_attention[t][a](x['features_%s' %(a)]) for a in self.auxilary_tasks if a!= t} for t in self.tasks}
        out = {t: x['features_%s' %(t)] + torch.sum(torch.stack([v for v in adapters[t].values()]), dim=0) for t in self.tasks}
        return out
