import numpy as np
from Traditional.Image import Image
import collections
import cv2
from sklearn.externals import joblib
import matplotlib
matplotlib.rcParams['figure.dpi']= 300
import matplotlib.pyplot as plt


def calculate_difference(p1, feature_vector):
    return np.sum(np.absolute(p1.get_feature_vector() - feature_vector)).astype(int)

def calculate_difference_std(p1, feature_vector):
    return np.std(np.absolute(p1.get_feature_vector() - feature_vector)).astype(int)


def get_best_images(classifier, p1):
    dict = {}

    image_category = int(np.asscalar(classifier.predict(p1.get_feature_vector().transpose())))

    count = image_category*100 + 1
    for i in range(100):
        filename = str(image_category) + "_" + str(count)
        p = np.load('featuredb/' + filename + ".npy")
        count = count + 1
        distance = calculate_difference(p1, p)

        dict[distance] = filename

    return collections.OrderedDict(sorted(dict.items()))

def show_top_images(p):
    plt.axis('off')
    plt.imshow(cv2.cvtColor(p.get_image_rgb(), cv2.COLOR_BGR2RGB))


    classifier = joblib.load('random_forest.pkl')
    dict = get_best_images(classifier, p)

    fig = plt.figure(figsize=(10, 10))
    columns = 4
    rows = 5

    for i in range(1, columns * rows + 1):
        path = r"C:\Users\daanv\PycharmProjects\imageretrieval\Corel100" + "\\" + dict[list(dict)[i]] + ".jpg"
        img = cv2.imread(path, 1)
        fig.add_subplot(rows, columns, i)

        plt.axis('off')
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    plt.show()

p = Image(43, 76)
show_top_images(p)