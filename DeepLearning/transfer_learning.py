import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn
import glob
from PIL import Image
import re
import os

imgSize = 255
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

path = "../Corel100/*.jpg"
layer = 'linear_1'
directoryName = layer+"_features"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.vgg19_bn(pretrained=True).eval()
classifier = nn.Sequential(
    nn.Linear(10,10)
)
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
            features = model.avgpool(model.features(image)).reshape(-1)
            feature = model.classifier[0](features)
            numpyFeature = features.detach().numpy()
            name = re.findall(pattern, file)[0]
            dict[name] = numpyFeature
        except Exception as ex:
            print("Could not make feature for image {} with exception {}".format(file, ex))
