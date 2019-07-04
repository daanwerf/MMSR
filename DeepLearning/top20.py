import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn
import glob
import numpy as np
import re
import pickle

with open('outputs.pkl', 'rb') as file:
    data = pickle.load(file)

labelsInterest = {56: [], 4: [], 34: [], 16: [], 45: [], 61: [], 92: [], 90: [], 6: [], 55: []}
for row in data:
    # row[2] contains predicted label
    labelsInterest[row[2]].append(row)

top20 = {}
for key, values in labelsInterest.items():
    searchSet = values[:5]
    for search in searchSet:
        distances = []
        for others in values:
            if others[0] != search[0]:
                distance = sum(torch.abs(search[3] - others[3])).tolist()[0]
                distances.append([distance, others])
        sortedDistances = sorted(distances, key=lambda x: x[0])
        top20[search[0]] = sortedDistances[:20]

with open("top20.pkl", 'wb') as file:
    pickle.dump(top20, file)

print("Done")

