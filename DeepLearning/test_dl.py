import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn
import glob
import numpy as np
import re
import pickle

classes = 100


def getScores(predicted, testLabels):
    correctArray = [0 for i in range(classes)]
    predictionsClass = [0 for i in range(classes)]
    for i in range(len(testLabels)):
        predictionsClass[int(predicted[i])] += 1
        if predicted[i] == testLabels[i]:
            correctArray[int(predicted[i])] += 1
    # Assume we have same amount of test objects for each class
    positives = len(testLabels) / classes
    return {"precision": [correctArray[i] / predictionsClass[i] for i in range(len(correctArray))],
            "recall": [correctArray[i] / positives for i in range(len(correctArray))]}


path = "db_features/*.npy"
pattern = "[0-9]+"
data = []
for index, file in enumerate(glob.glob(path)):
    label = int(re.findall(pattern, file)[0])
    data.append([np.load(file), label])
featureDimension = 25088
with open('validation.pkl', 'rb') as file:
    test = pickle.load(file)
print(len(test))
testloader = torch.utils.data.DataLoader(test, shuffle=True, batch_size=1)
testloader = torch.utils.data.DataLoader(data, shuffle=True, batch_size=1)
classifier = nn.Sequential(
    nn.Linear(featureDimension, 4096),
    nn.ReLU(inplace=True),
    nn.Dropout(p=0.5),
    nn.Linear(4096, 4096),
    nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(4096, 100)
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classifier.to(device)
classifierState = torch.load('model2')
classifier.load_state_dict(classifierState)

errors = 0
total = 0
labelArray = []
predictedArray = []
with torch.no_grad():
    for data in testloader:
        images, labels = data
        if torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()
        outputs = classifier(images)
        _, predicted = torch.max(outputs.data, 1)
        labelArray.append(labels.tolist()[0])
        predictedArray.append(predicted.tolist()[0])
        total += labels.size(0)
        errors += (predicted != labels).sum().item()

print("error rate is {}".format(errors / total))
score = getScores(predictedArray, labelArray)
print("Average Precision is {} and Average Recall is {}".format(np.mean(score['precision']), np.mean(score['recall'])))