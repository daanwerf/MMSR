from sklearn.externals import joblib
import glob
import numpy as np
import re
from sklearn.model_selection import cross_validate

classes = 100
classifier = joblib.load("random_forest.pkl")
features = np.zeros((10000, 768))
labels = np.zeros(10000)
testSplit = 0.1

