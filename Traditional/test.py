import numpy as np
from Traditional.Image import Image
import collections
import cv2
from pprint import pprint


def calculate_difference(p1, feature_vector):
    print(feature_vector.shape)
    return np.sum(np.absolute(p1.get_feature_vector() - feature_vector)).astype(int)

def calculate_difference_std(p1, feature_vector):
    return np.std(np.absolute(p1.get_feature_vector() - feature_vector)).astype(int)


def get_best_images(p1):
    dict = {}
    heap = []
    k = 10  # maximum amount of retrieved images

    count = 1
    for i in range(100):
        print(str(i) + "%")
        for j in range(100):
            filename = str(i) + "_" + str(count)
            p = np.load('featuredb/' + filename + ".npy")
            count = count + 1
            distance = calculate_difference(p1, p)

            dict[distance] = filename

    return collections.OrderedDict(sorted(dict.items()))

def show_top_10_images(dict):
    for i in range(len(dict)):
        path = r"C:\Users\daanv\PycharmProjects\imageretrieval\Corel100" + "\\" + dict[list(dict)[i]] + ".jpg"
        img = cv2.imread(path, 1)
        cv2.imshow('Image ' + str(i), img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        if i == 10:
            break

p1 = Image(9,36)

pprint(load_data())

#d = get_best_images(p1)
#show_top_10_images(d)

