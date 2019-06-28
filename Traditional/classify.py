from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate, train_test_split
import numpy as np
import glob
import re
import pickle

trainSplit = 0.9
classes = 100
samplesClass = 100
def getValidationScore(classifier, features, labels):
    classifier.fit(features, labels)
    scoring = {'recall_macro', 'precision_macro'}
    return cross_validate(classifier, features, labels, cv=5, scoring=scoring)

def getTestLabels(classifier, train, trainLabel, test):
    classifier.fit(train, trainLabel)
    return classifier.predict(test)

def getScores(predicted, testLabels):
    correctArray = [0 for i in range(classes)]
    predictionsClass = [0 for i in range(classes)]
    for i in range(len(testLabel)):
        predictionsClass[int(predicted[i])] += 1
        if predicted[i] == testLabels[i]:
            correctArray[int(predicted[i])] += 1
    return {"precision" : [correctArray[i] / predictionsClass[i] for i in correctArray],
            "recall" : [correctArray[i] / 100*trainSplit for i in correctArray]}

features = np.zeros((10000, 768))
labels = np.zeros(10000)
for index, file in enumerate(glob.glob("featuredb/*.npy")):
    features[index] = np.load(file).T
    labels[index] = int(re.findall('[0-9]+', file)[0])

runs = 10
scores = {"precision" : 0, "recall" : 0}
for run in range(runs):
    print("Run {}".format(run))
    # Fair splitting can be done more efficient.
    trainIndices = np.array([np.random.choice(np.arange(i*classes, i*classes+100), replace=False, size=int(100*trainSplit)) for i in range(classes)]).flatten()
    testIndices = np.setdiff1d(np.arange(10000), trainIndices)
    train = features[trainIndices]
    trainLabel = labels[trainIndices]
    test = features[testIndices]
    testLabel = labels[testIndices]

    models = [RandomForestClassifier(n_estimators=128, random_state=0)]
    names = ["Random_Forest"]
    scaler = StandardScaler().fit(features)

    # TODO save best model
    save = True
    for index, model in enumerate(models):
        predicted = getTestLabels(model,scaler.transform(train),trainLabel, scaler.transform(test))
        if save: pickle.dump(model, open(names[index], 'wb'))
        score = getScores(predicted, testLabel)
        scores['precision'] += np.mean(score['precision'])
        scores['recall'] += np.mean(score['recall'])

print("Average precision is {} and average recall is {}".
      format(scores['precision'] / runs, scores['recall'] / runs))




