import numpy as np
from Traditional.Image import Image
import collections
import cv2
from sklearn.externals import joblib
import matplotlib
matplotlib.rcParams['figure.dpi']= 300
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


def calculate_difference(p1, feature_vector):
    return np.sum(np.absolute(p1.get_feature_vector() - feature_vector)).astype(int)


def get_best_images(classifier, image):
    dict = {}

    i_category = int(np.asscalar(classifier.predict(image.get_feature_vector().transpose())))

    for i in range(100):
        print(str(i) + "%")
        for j in range(1, 101):
            image2 = Image(i, j)
            i2_feature_vector = image2.get_feature_vector()
            i2_category = int(np.asscalar(classifier.predict(i2_feature_vector.transpose())))

            if i2_category == i_category:
                distance = calculate_difference(image, i2_feature_vector)
                dict[distance] = image2.get_file_name()

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

p = Image(6, 21)
show_top_images(p)