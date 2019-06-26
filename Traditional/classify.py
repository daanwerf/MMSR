from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.model_selection import cross_validate
import numpy as np
import glob
import re

def getScore(classifier, features, labels):
    classifier.fit(features, labels)
    scoring = {'recall_macro', 'precision_macro'}
    return cross_validate(classifier, features, labels, cv=5, scoring=scoring)

features = []
labels = []
count = 0
samples = 100*15
for file in glob.glob("featuredb/*.npy"):
    if samples == count:
        break
    labels.append(re.findall('[0-9]+', file)[0])
    features.append(np.load(file).T)
    count += 1


features = np.array(features).reshape(samples, features[0].shape[1])
# TODO more models
# TODO grid search for hyperparameters svc
models = [LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial'), svm.SVC(gamma='scale', decision_function_shape='ovo'), KNeighborsClassifier(n_neighbors=3)]
for model in models:
    score = getScore(model,features,labels)
    print("Precision is {}".format(score["test_precision_macro"]))
    print("Recall is {} \n".format(score["test_recall_macro"]))


