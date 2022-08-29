import numpy as np
import torch
'''
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import vgg16
import tensorflow.keras.applications as models
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications.imagenet_utils import decode_predictions
from tensorflow.keras.preprocessing.image import img_to_array
'''
import random
from random import shuffle
import torch
from torchvision import models
import torchvision.transforms as transforms
import torchvision.datasets as dsets

import torch.nn as nn
import math
from collections import OrderedDict
from torch.utils import model_zoo
import os 

from pretrained.cifar10.vgg import vgg16_bn, vgg19_bn
from pretrained.cifar10.resnet import resnet50

use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")
torch.backends.cudnn.deterministic = True

# Image processing
#------------------------------------------------------------------------------------------------------------------------
def _01(im):
    min_im = np.min(im)
    max_im = np.max(im)
    _01 = (im-min_im)/(max_im-min_im)
    return _01 

def rescale_linear(array, new_min, new_max):
    """Rescale an arrary linearly."""
    minimum, maximum = np.min(array), np.max(array)
    m = (new_max - new_min) / (maximum - minimum)
    b = new_min - m * minimum
    return m * array + b

prediction_preprocess = transforms.Compose([
    transforms.ToTensor(), # ToTensor : [0, 255] -> [0, 1], (224,224,3) -> (3,224,224)   
])

# Others - experiments
#------------------------------------------------------------------------------------------------------------------------------
# combinations for patch replacement
def create_comb(patch_size,half):
    comb = np.array(list(itertools.product(range(half,224,patch_size),range(half,224,patch_size))))
    return comb	

# combinations for shuffling
def create_shuffle_comb(shuffle_size,half):
    comb1=np.array(list(itertools.product(range(half,224,shuffle_size),range(half,224,shuffle_size))))
    comb2=np.array(list(itertools.product(range(half,224,shuffle_size),range(half,224,shuffle_size))))
    np.random.shuffle(comb1)
    np.random.shuffle(comb2)
    return comb1, comb2

def shuffle(im, comb1, comb2, half):
    locs = []
    shuffled_im = np.zeros((3,224,224))
    for x in range(len(comb1)): 
        bbox_centre1_ij = comb1[x]
        bbox_centre2_ij = comb2[x]
        loc_i = bbox_centre1_ij[0]
        loc_j = bbox_centre1_ij[1]
        loc_i2 = bbox_centre2_ij[0]
        loc_j2 = bbox_centre2_ij[1]
        shuffled_im[:,loc_i2-half:loc_i2+half,loc_j2-half:loc_j2+half] = im[:,loc_i-half:loc_i+half,loc_j-half:loc_j+half]
    return shuffled_im

# Prediction
#---------------------------------------------------------------------------------------------------------------------------------
def softmax(x): 
    e_x = np.exp(x - np.max(x)) 
    return e_x / e_x.sum()

# Model creation
#-------------------------------------------------------------------------------------------------------------------------
class Normalize(nn.Module) :
    def __init__(self, mean, std) :
        super(Normalize, self).__init__()
        self.register_buffer('mean', torch.Tensor(mean))
        self.register_buffer('std', torch.Tensor(std))
        
    def forward(self, input):
        # Broadcasting
        mean = self.mean.reshape(1, 3, 1, 1)
        std = self.std.reshape(1, 3, 1, 1)
        return (input - mean) / std

norm_layer = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

def create_torchmodel_imagenet(network_name):
    if network_name == 'BagNet9_simple':
        with torch.no_grad():
            model = bagnet9().to(device)
            model=model.eval()
    elif network_name == 'BagNet17_simple':
        with torch.no_grad():
            model = bagnet17().to(device)
            model=model.eval()
    elif network_name == 'BagNet33_simple':
        with torch.no_grad():
            model = bagnet33().to(device)
            model=model.eval()
    elif network_name == 'BagNet9':
        with torch.no_grad():
            model = nn.Sequential(
                norm_layer,
                bagnet9()
            ).to(device)
            model=model.eval()
    elif network_name == 'BagNet17':
        with torch.no_grad():
            model = nn.Sequential(
                norm_layer,
                bagnet17()
            ).to(device)
            model=model.eval()
    elif network_name == 'BagNet33':
        with torch.no_grad():
            model = nn.Sequential(
                norm_layer,
                bagnet33()
            ).to(device)
            model=model.eval()
    elif network_name == 'VGG16':
        with torch.no_grad():
            model = nn.Sequential(
                norm_layer,
                models.vgg16(pretrained=True)
            ).to(device)
            model=model.eval()
    elif network_name == 'VGG19':
        with torch.no_grad():
            model = nn.Sequential(
                norm_layer,
                models.vgg19(pretrained=True)
            ).to(device)
            model=model.eval()
    elif network_name == 'ResNet50':
        with torch.no_grad():
            model = nn.Sequential(
                norm_layer,
                models.resnet50(pretrained=True)
            ).to(device)
            model=model.eval()
    elif network_name == 'ResNet101':
        with torch.no_grad():
            model = nn.Sequential(
                norm_layer,
                models.resnet101(pretrained=True)
            ).to(device)
            model=model.eval()
    elif network_name == 'ResNet152':
        with torch.no_grad():
            model = nn.Sequential(
                norm_layer,
                models.resnet152(pretrained=True)
            ).to(device)
            model=model.eval()
    elif network_name == 'DenseNet121':
        with torch.no_grad():
            model = nn.Sequential(
                norm_layer,
                models.densenet121(pretrained=True)
            ).to(device)
            model=model.eval()
    elif network_name == 'DenseNet169':
        with torch.no_grad():
            model = nn.Sequential(
                norm_layer,
                models.densenet169(pretrained=True)
            ).to(device)
            model=model.eval()
    elif network_name == 'DenseNet201':
        with torch.no_grad():
            model = nn.Sequential(
                norm_layer,
                models.densenet201(pretrained=True)
            ).to(device)
            model=model.eval()
    elif network_name == 'MobileNet':
        with torch.no_grad():
            model = nn.Sequential(
                norm_layer,
                models.mobilenet_v2(pretrained=True)
            ).to(device)
            model=model.eval()
    elif network_name == 'MNASNet':
        with torch.no_grad():
            model = nn.Sequential(
                norm_layer,
                models.mnasnet1_0(pretrained=True)
            ).to(device)
            model=model.eval()
    elif network_name == 'InceptionV3':
        model = nn.Sequential(
            norm_layer,
            models.inception_v3(pretrained=True)
        ).to(device)
        model=model.eval()
    elif network_name == 'ResNet50_SIN':
        model_urls = {
            'ResNet50_SIN': 'https://bitbucket.org/robert_geirhos/texture-vs-shape-pretrained-models/raw/6f41d2e86fc60566f78de64ecff35cc61eb6436f/resnet50_train_60_epochs-c8e5653e.pth.tar',
            'alexnet_trained_on_SIN': 'https://bitbucket.org/robert_geirhos/texture-vs-shape-pretrained-models/raw/0008049cd10f74a944c6d5e90d4639927f8620ae/alexnet_train_60_epochs_lr0.001-b4aa5238.pth.tar',
        }
        print("Using the ResNet50 architecture.")
        loaded_model = models.resnet50(pretrained=False)
        loaded_model = torch.nn.DataParallel(loaded_model).cuda()
        checkpoint = model_zoo.load_url(model_urls[network_name])
        loaded_model.load_state_dict(checkpoint["state_dict"])
        with torch.no_grad():
           model = nn.Sequential(
              norm_layer,
              loaded_model
           ).to(device)
           model = model.eval()
    return model

norm_layer_cifar10 = Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2471, 0.2435, 0.2616])

def create_torchmodel_cifar10(network_name):
    if network_name == 'VGG16':
        with torch.no_grad():
            loaded_model = vgg16_bn()
            loaded_model.load_state_dict(torch.load('pretrained/cifar10/{}_bn.pt'.format(network_name.lower())))
            model = nn.Sequential(
                norm_layer_cifar10,
                loaded_model
            ).to(device)
            model=model.eval()
    elif network_name == 'VGG19':
        with torch.no_grad():
            model = nn.Sequential(
                norm_layer,
                models.vgg19(pretrained=True)
            ).to(device)
            model=model.eval()
    if network_name == 'ResNet50':
        with torch.no_grad():
            loaded_model = resnet50()
            loaded_model.load_state_dict(torch.load('pretrained/cifar10/{}.pt'.format(network_name.lower())))
            model = nn.Sequential(
                norm_layer_cifar10,
                loaded_model
            ).to(device)
            model=model.eval()
    elif network_name == 'ResNet101':
        with torch.no_grad():
            model = nn.Sequential(
                norm_layer,
                models.resnet101(pretrained=True)
            ).to(device)
            model=model.eval()
    elif network_name == 'ResNet152':
        with torch.no_grad():
            model = nn.Sequential(
                norm_layer,
                models.resnet152(pretrained=True)
            ).to(device)
            model=model.eval()
    elif network_name == 'DenseNet121':
        with torch.no_grad():
            model = nn.Sequential(
                norm_layer,
                models.densenet121(pretrained=True)
            ).to(device)
            model=model.eval()
    elif network_name == 'DenseNet169':
        with torch.no_grad():
            model = nn.Sequential(
                norm_layer,
                models.densenet169(pretrained=True)
            ).to(device)
            model=model.eval()
    elif network_name == 'DenseNet201':
        with torch.no_grad():
            model = nn.Sequential(
                norm_layer,
                models.densenet201(pretrained=True)
            ).to(device)
            model=model.eval()
    elif network_name == 'MobileNet':
        with torch.no_grad():
            model = nn.Sequential(
                norm_layer,
                models.mobilenet_v2(pretrained=True)
            ).to(device)
            model=model.eval()
    elif network_name == 'MNASNet':
        with torch.no_grad():
            model = nn.Sequential(
                norm_layer,
                models.mnasnet1_0(pretrained=True)
            ).to(device)
            model=model.eval()
    return model


# BagNet
#-----------------------------------------------------------------------------------------------------------------------------------
__all__ = ['bagnet9', 'bagnet17', 'bagnet33']

model_urls = {
             'bagnet9': 'https://bitbucket.org/wielandbrendel/bag-of-feature-pretrained-models/raw/d413271344758455ac086992beb579e256447839/bagnet8.h5',
             'bagnet17': 'https://bitbucket.org/wielandbrendel/bag-of-feature-pretrained-models/raw/d413271344758455ac086992beb579e256447839/bagnet16.h5',
             'bagnet33': 'https://bitbucket.org/wielandbrendel/bag-of-feature-pretrained-models/raw/d413271344758455ac086992beb579e256447839/bagnet32.h5',
             }

def bagnet9():
	model_path = keras.utils.get_file(
	                'bagnet8.h5',
	                model_urls['bagnet9'],
	                cache_subdir='models',
	                file_hash='5b70adc7c4ff77d932dbba485a5ea1d333a65e777a45511010f22e304a2fdd69')

	return load_model(model_path)

def bagnet17():
	model_path = keras.utils.get_file(
	                'bagnet16.h5',
	                model_urls['bagnet17'],
	                cache_subdir='models',
	                file_hash='b262dfee15a86c91e6aa21bfd86505ecd20a539f7f7c72439d5b1d352dd98a1d')

	return load_model(model_path)

def bagnet33(pretrained=True, strides=[2, 2, 2, 1], **kwargs):
    """Constructs a Bagnet-33 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = BagNet(Bottleneck, [3, 4, 6, 3], strides=strides, kernel3=[1,1,1,1], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['bagnet33'], map_location = 'cpu'))
    return model

#pytorch
import torch.nn as nn
import math
import torch
from collections import OrderedDict
from torch.utils import model_zoo

import os 
#dir_path = os.path.dirname(os.path.realpath(__file__))

__all__ = ['bagnet9', 'bagnet17', 'bagnet33']

model_urls = {
            'bagnet9': 'https://bitbucket.org/wielandbrendel/bag-of-feature-pretrained-models/raw/249e8fa82c0913623a807d9d35eeab9da7dcc2a8/bagnet8-34f4ccd2.pth.tar',
            'bagnet17': 'https://bitbucket.org/wielandbrendel/bag-of-feature-pretrained-models/raw/249e8fa82c0913623a807d9d35eeab9da7dcc2a8/bagnet16-105524de.pth.tar',
            'bagnet33': 'https://bitbucket.org/wielandbrendel/bag-of-feature-pretrained-models/raw/249e8fa82c0913623a807d9d35eeab9da7dcc2a8/bagnet32-2ddd53ed.pth.tar',
                            }


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, kernel_size=1):
        super(Bottleneck, self).__init__()
        # print('Creating bottleneck with kernel size {} and stride {} with padding {}'.format(kernel_size, stride, (kernel_size - 1) // 2))
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=kernel_size, stride=stride,
                               padding=0, bias=False) # changed padding from (kernel_size - 1) // 2
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, **kwargs):
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
        
        if residual.size(-1) != out.size(-1):
            diff = residual.size(-1) - out.size(-1)
            residual = residual[:,:,:-diff,:-diff]
        
        out += residual
        out = self.relu(out)

        return out


class BagNet(nn.Module):

    def __init__(self, block, layers, strides=[1, 2, 2, 2], kernel3=[0, 0, 0, 0], num_classes=1000, avg_pool=True):
        self.inplanes = 64
        super(BagNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=1, stride=1, padding=0,
                               bias=False)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=0.001)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=strides[0], kernel3=kernel3[0], prefix='layer1')
        self.layer2 = self._make_layer(block, 128, layers[1], stride=strides[1], kernel3=kernel3[1], prefix='layer2')
        self.layer3 = self._make_layer(block, 256, layers[2], stride=strides[2], kernel3=kernel3[2], prefix='layer3')
        self.layer4 = self._make_layer(block, 512, layers[3], stride=strides[3], kernel3=kernel3[3], prefix='layer4')
        self.avgpool = nn.AvgPool2d(1, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.avg_pool = avg_pool
        self.block = block

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, kernel3=0, prefix=''):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        kernel = 1 if kernel3 == 0 else 3
        layers.append(block(self.inplanes, planes, stride, downsample, kernel_size=kernel))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            kernel = 1 if kernel3 <= i else 3
            layers.append(block(self.inplanes, planes, kernel_size=kernel))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.avg_pool:
            x = nn.AvgPool2d(x.size()[2], stride=1)(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
        else:
            x = x.permute(0,2,3,1)
            x = self.fc(x)

        return x

def bagnet33(pretrained=True, strides=[2, 2, 2, 1], **kwargs):
    """Constructs a Bagnet-33 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = BagNet(Bottleneck, [3, 4, 6, 3], strides=strides, kernel3=[1,1,1,1], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['bagnet33']))
    return model

def bagnet17(pretrained=True, strides=[2, 2, 2, 1], **kwargs):
    """Constructs a Bagnet-17 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = BagNet(Bottleneck, [3, 4, 6, 3], strides=strides, kernel3=[1,1,1,0], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['bagnet17']))
    return model

def bagnet9(pretrained=True, strides=[2, 2, 2, 1], **kwargs):
    """Constructs a Bagnet-9 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = BagNet(Bottleneck, [3, 4, 6, 3], strides=strides, kernel3=[1,1,0,0], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['bagnet9']))
    return model