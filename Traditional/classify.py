from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate
import numpy as np
import glob
import re
from sklearn.externals import joblib

trainSplit = 0.8
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
    for i in range(len(testLabels)):
        predictionsClass[int(predicted[i])] += 1
        if predicted[i] == testLabels[i]:
            correctArray[int(predicted[i])] += 1
    # Assume we have same amount of test objects for each class
    positives = len(testLabels) / classes
    return {"precision" : [correctArray[i] / predictionsClass[i] for i in range(len(correctArray))],
            "recall" : [correctArray[i] / positives for i in range(len(correctArray))]}
def loadModel(filename):
    classifier = None
    with open(filename, 'rb'):
        classifier = joblib.load(filename)
    return classifier

features = np.zeros((10000, 768))
labels = np.zeros(10000)
for index, file in enumerate(glob.glob("featuredb/*.npy")):
    features[index] = np.load(file).T
    labels[index] = int(re.findall('[0-9]+', file)[0])

runs = 10
scores = {"precision" : 0, "recall" : 0}
bestPerformance = 2**32
for run in range(runs):
    print("Run {}".format(run))
    # Fair splitting can be done more efficient.
    trainIndices = np.array([np.random.choice(np.arange(i*classes, i*classes+100), replace=False, size=int(100*trainSplit)) for i in range(classes)]).flatten()
    testIndices = np.setdiff1d(np.arange(10000), trainIndices)
    train = features[trainIndices]
    trainLabel = labels[trainIndices]
    test = features[testIndices]
    testLabel = labels[testIndices]
    # More estimator is generally better, but it becomes slower and some point it isnt worth it anymore to expand the estimators
    models = [RandomForestClassifier(n_estimators=256, random_state=0)]
    names = ["random_forest"]
    for index, model in enumerate(models):
        predicted = getTestLabels(model,train,trainLabel, test)
        score = getScores(predicted, testLabel)
        scores['precision'] += np.mean(score['precision'])
        scores['recall'] += np.mean(score['recall'])
        if bestPerformance > (np.mean(score['precision'])+ np.mean(score['recall'])):
            print("Save best model")
            joblib.dump(model, "{}.pkl".format(names[index]))
            bestPerformance = (np.mean(score['precision']) + np.mean(score['recall']))

print("Average precision is {} and average recall is {}".
      format(scores['precision'] / runs, scores['recall'] / runs))




