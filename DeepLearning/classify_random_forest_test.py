from sklearn.externals import joblib
import glob
import numpy as np
import re
from sklearn.model_selection import cross_validate

classes = 100
classifier = joblib.load("random_forest2.pkl")
features = np.zeros((10000, 25088))
labels = np.zeros(10000)
testSplit = 0.1

