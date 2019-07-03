import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn
from torch.autograd import Variable
import glob
from PIL import Image
import re
import os
import numpy as np

imgSize = 244

path = "db_features/*.npy"
pattern = "[0-9]+"
featureDimension = 25088
train = []
for index, file in enumerate(glob.glob(path)):
    label = int(re.findall(pattern, file)[0])
    train.append([np.load(file), label])
    if index == 100:
        break

classifier = nn.Sequential(
    nn.Linear(featureDimension, 4096),
    nn.ReLU(inplace=True),
    nn.Dropout(p=0.5),
    nn.Linear(4096, 4096),
    nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(4096, 100)
)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(classifier.parameters(),lr=0.1,weight_decay=1e-4)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classifier.to(device)
trainloader = torch.utils.data.DataLoader(train, shuffle=True, batch_size=4)

epochs = 2
for epoch in range(epochs):
    runningLoss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            labels = labels.cuda()
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = classifier(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        runningLoss += loss.item()
        if i % 100 == 0:  # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, runningLoss / 100))
            runningLoss = 0.0