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
import pickle

def testModel(model, test):
    errors = 0
    total = 0
    with torch.no_grad():
        for data in test:
            images, labels = data
            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            errors += (predicted != labels).sum().item()
    print("error rate is {}".format(errors / total))

imgSize = 244
classes = 100
featureDimension = 25088
# Fair splitting can be done more efficient.
trainSplit = 0.7
trainIndices = np.array([np.random.choice(np.arange(i*classes, i*classes+100), replace=False, size=int(100*trainSplit)) for i in range(classes)]).flatten()
testIndices = np.setdiff1d(np.arange(10000), trainIndices)
# Remove the image index for failed image
if 9999 in trainIndices:
    trainIndices = np.delete(trainIndices, np.argwhere(trainIndices == 9999))
else:
    testIndices = np.delete(testIndices, np.argwhere(testIndices == 9999))
    
path = "db_features/*.npy"
pattern = "[0-9]+"
train = []
test = []
for index, file in enumerate(glob.glob(path)):
    label = int(re.findall(pattern, file)[0])
    if index in trainIndices:
        train.append([np.load(file), label])
    else:
        test.append([np.load(file), label])       
with open('validation.pkl', 'wb') as file:
    pickle.dump(test, file)      
# Already shuffled by random.choice
trainloader = torch.utils.data.DataLoader(train, shuffle=False, batch_size=16)
testloader = torch.utils.data.DataLoader(test, shuffle=False, batch_size=1)
    
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
optimizer = torch.optim.Adam(classifier.parameters(),lr=0.0001)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classifier.to(device)
epochs = 10
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
        if i % 500 == 0:  # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, runningLoss / 500))
            runningLoss = 0.0
            
    print("epoch {} done".format(epoch))
    testModel(classifier, testloader)
    torch.save(classifier.state_dict(), "model{}".format(epoch))