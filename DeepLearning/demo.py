import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn
import glob
import numpy as np
import re
import pickle
import matplotlib.pyplot as plt
from PIL import Image

with open('top20.pkl', 'rb') as file:
    data = pickle.load(file)

path = "../Corel100/"
pattern = "[0-9]+"
labelsInterest = {56: 0, 4: 0, 34: 0, 16: 0, 45: 0, 61: 0, 92: 0, 90: 0, 6: 0, 55: 0}
recall = {56: 0, 4: 0, 34: 0, 16: 0, 45: 0, 61: 0, 92: 0, 90: 0, 6: 0, 55: 0}
for key, values in data.items():
    correct = 0
    label = int(re.findall(pattern, key)[0])
    for row in values:
        if row[1][1] == row[1][2]:
            correct += 1
    labelsInterest[label] += correct

for key,val in precision.items():
    print("Class is {} and recall is {}".format(recall))

pattern = "[0-9]+_[0-9]+"
imIndex = 0
pattern2 = "[0-9]+"
for key, value in data.items():
    label = int(re.findall(pattern2, key)[0])
    if label == 6:
        imageInfo = re.findall(pattern, key)[0]
        fig = plt.figure()
        with Image.open("{}{}.jpg".format(path,imageInfo)) as image:
            plt.imshow(image)
            plt.axis('off')
            plt.savefig("output/{}search.jpg".format(imIndex))
            plt.close(fig)

        fig = plt.figure()
        for index, item in enumerate(value):
            imageInfo = re.findall(pattern, item[1][0])[0]
            with Image.open("{}{}.jpg".format(path,imageInfo)) as image:
                plt.subplot(5, 4,index+1)
                plt.axis('off')
                plt.text(0, 0, "{0:.4f}".format(item[0]))
                plt.imshow(image)
        plt.savefig("output/{}top20.jpg".format(imIndex))
        plt.close(fig)
        imIndex+=1
        if imIndex == 5:
            break




