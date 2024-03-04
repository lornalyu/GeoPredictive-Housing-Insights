import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import sampler, RandomSampler, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
from PIL import Image, ImageOps
import torchvision.datasets as dset
import torchvision.transforms as T

import numpy as np
from sklearn.metrics import confusion_matrix
import time

import seaborn as sns
#from __future__ import print_function 
#from __future__ import division
import matplotlib.pyplot as plt
import time
import os
import copy

# see more at: https://github.com/kkraoj/damaged_structures_detector/blob/master/codes/Final%20Project.ipynb

# Below is a copy and paste from shared git damaged_structures_detector !!!


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(model_name, num_classes, width, channels, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet18
        """                
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

        if(width != input_size):       
          W1 = width
          if (input_size > W1):
            F = 1
            P = int((input_size - W1) / 2)
          else:
            P = 0
            F = W1 - input_size +1
          
          first_conv_layer = nn.Conv2d(channels, 3, kernel_size=F, stride=1, padding=P, dilation=1, groups=1, bias=True)
          model_ft = nn.Sequential(first_conv_layer, model_ft)
  

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

        if(width != input_size):       
          W1 = width
          if (input_size > W1):
            F = 1
            P = int((input_size - W1) / 2)
          else:
            P = 0
            F = W1 - input_size +1
          
          first_conv_layer = [nn.Conv2d(channels, 3, kernel_size=F, stride=1, padding=P, dilation=1, groups=1, bias=True)]
          first_conv_layer.extend(list(model_ft.features))  
          model_ft.features= nn.Sequential(*first_conv_layer)

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

        if(width != input_size):       
          W1 = width
          if (input_size > W1):
            F = 1
            P = int((input_size - W1) / 2)
          else:
            P = 0
            F = W1 - input_size +1
          
          first_conv_layer = [nn.Conv2d(channels, 3, kernel_size=F, stride=1, padding=P, dilation=1, groups=1, bias=True)]
          first_conv_layer.extend(list(model_ft.features))  
          model_ft.features= nn.Sequential(*first_conv_layer)

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

        if(width != input_size):       
          W1 = width
          if (input_size > W1):
            F = 1
            P = int((input_size - W1) / 2)
          else:
            P = 0
            F = W1 - input_size +1
          
          first_conv_layer = [nn.Conv2d(channels, 3, kernel_size=F, stride=1, padding=P, dilation=1, groups=1, bias=True)]
          first_conv_layer.extend(list(model_ft.features))  
          model_ft.features= nn.Sequential(*first_conv_layer)
        

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes) 
        input_size = 224

        if(width != input_size):       
          W1 = width
          if (input_size > W1):
            F = 1
            P = int((input_size - W1) / 2)
          else:
            P = 0
            F = W1 - input_size +1
          
          first_conv_layer = [nn.Conv2d(channels, 3, kernel_size=F, stride=1, padding=P, dilation=1, groups=1, bias=True)]
          first_conv_layer.extend(list(model_ft.features))  
          model_ft.features= nn.Sequential(*first_conv_layer)

    elif model_name == "inception":
        """ Inception v3 
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299

        if(width != input_size):       
          W1 = width
          if (input_size > W1):
            F = 1
            P = int((input_size - W1) / 2)
          else:
            P = 0
            F = W1 - input_size +1
          
          first_conv_layer = nn.Conv2d(channels, 3, kernel_size=F, stride=1, padding=P, dilation=1, groups=1, bias=True)
          model_ft.AuxLogits = nn.Sequential(first_conv_layer, model_ft.AuxLogits)
          model_ft = nn.Sequential(first_conv_layer, model_ft)


    else:
        print("Invalid model name, exiting...")
        exit()
    
    return model_ft, input_size



# Intended Input Image Size
HEIGHT = 224
WIDTH = 224
CHANNELS = 3

# Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
MODEL_NAME = "resnet"
# Number of classes in the dataset
NUM_CLASSES = 2
# Flag for feature extracting. When False, we finetune the whole model, 
#   when True we only update the reshaped layer params
FEATURE_EXTRACT = False

# Initialize the model for this run

model_ft, input_size = initialize_model(MODEL_NAME, NUM_CLASSES, WIDTH, CHANNELS, FEATURE_EXTRACT, use_pretrained=True)

# Print the parameters of the model we just instantiated
#for param in model_ft.parameters():
#  print(param.data)