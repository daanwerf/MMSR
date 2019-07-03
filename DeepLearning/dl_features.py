import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn
from torch.autograd import Variable
import glob
import copy
from PIL import Image
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import re
import pickle
import os

def getModel(cnn, lastLayer):
    model = nn.Sequential()
    indexCov = 0
    indexLin = 0
    name = ""
    for module in cnn.children():
        if isinstance(module, nn.Sequential):
            for layer in module:
                if isinstance(layer, nn.Conv2d):
                    indexCov += 1
                    name = 'conv_{}'.format(indexCov)
                elif isinstance(layer, nn.ReLU):
                    name = 'relu_{}'.format(indexCov)
                    layer = nn.ReLU(inplace=False)
                elif isinstance(layer, nn.MaxPool2d):
                    name = 'pool_{}'.format(indexCov)
                elif isinstance(layer, nn.BatchNorm2d):
                    name = 'bn_{}'.format(indexCov)
                elif isinstance(layer, nn.Sequential):
                    name = 'seq_{}'.format(indexCov)
                elif isinstance(layer, nn.AdaptiveAvgPool2d):
                    name = 'avgpool_{}'.format(indexCov)
                elif isinstance(layer, nn.Linear):
                    indexLin += 1
                    name = 'linear_{}'.format(indexLin)
                model.add_module(name, layer)
                if name == lastLayer:
                    return  model
        if isinstance(module, nn.AdaptiveAvgPool2d):
            # Between convulution module and linear module is a between layer
            # Only add if you want to go the the linear module
            model.add_module("avgpool_", module)

    return model

imgSize = 224
classes = 10
transform = transforms.Compose([
    transforms.Resize((imgSize,imgSize)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])
def imToTrain(im):
    im = transform(im)
    # Set it into a new batch dimension
    im = im.unsqueeze(0)
    return im

path = "D:/Semester2/MMSR/Corel100/*.jpg"
layer = 'linear_1'
directoryName = layer+"_features"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.vgg19_bn(pretrained=True).eval()
#model = getModel(pretrained, layer)
model.to(device)
try:
    os.makedirs(directoryName)
except:
    print("Directory {} already exists".format(layer))
images = 0
pattern = "[0-9]+_[0-9]+"
for file in glob.glob(path):
    with Image.open(file) as image:
        try:
            image = imToTrain(image)
            if torch.cuda.is_available():
                image = image.cuda()
            features = model.avgpool(model.features((image))).reshape(-1)
            numpyFeature = features.detach().cpu().numpy()
            name = re.findall(pattern, file)[0]
            np.save(name, numpyFeature)
        except Exception as ex:
            print("Could not make feature for image {} with exception {}".format(file, ex))

