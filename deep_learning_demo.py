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
import re


def calculate_difference(feature_vector1, feature_vector2):
    return np.sum(np.absolute(feature_vector1 - feature_vector2)).astype(int)


def get_best_images(classifier, image):
    dict = {}

    i_feature_vector = image.get_deep_learning_feature_vector_from_db()
    i_classifier_category = int(np.asscalar(classifier.predict(np.reshape(i_feature_vector, (-1, 1)).transpose())))

    for i in range(100):
        for j in range(1, 101):
            print(str(i) + "." + str(j) + "%")
            image2 = Image(i, j)

            if int(image2.get_image_id()) == 9145:
                print("skipped broken image")
                continue

            i2_feature_vector = np.reshape(image2.get_deep_learning_feature_vector_from_db(), (-1, 1))
            i2_classifier_category = int(np.asscalar(classifier.predict(i2_feature_vector.transpose())))

            if i2_classifier_category == i_classifier_category:
                distance = calculate_difference(i_feature_vector, i2_feature_vector)
                dict[distance] = image2.get_file_name()

    return collections.OrderedDict(sorted(dict.items()))


def show_top_images(p):
    plt.axis('off')
    plt.imshow(cv2.cvtColor(p.get_image_rgb(), cv2.COLOR_BGR2RGB))

    classifier = joblib.load('random_forest_dl.pkl')
    dict = get_best_images(classifier, p)
    print(len(dict))
    print(dict)

    fig = plt.figure(figsize=(10, 10))
    columns = 4
    rows = 5

    for i in range(1, columns * rows + 1):
        path = r"Corel100" + "\\" + dict[list(dict)[i]] + ".jpg"
        img = cv2.imread(path, 1)
        fig.add_subplot(rows, columns, i)

        plt.axis('off')
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    plt.show()


def calculate_precision_and_recall_for_category(category = 11, first_x_images = 20, image_amount = 5):
    classifier = joblib.load('random_forest_dl.pkl')

    retrieved_count = first_x_images*image_amount
    relevant_count = 0

    for k in range(0, image_amount):
        print("Run " + str(k + 1) + " of " + str(image_amount))

        image = Image(category, k+1)
        image_true_category = int(image.get_image_category())
        dict = get_best_images(classifier, image)

        for i in range(first_x_images):
            filename = dict[list(dict)[i]]
            result_image_category = int(re.split("_", filename)[0])

            if image_true_category == result_image_category:
                relevant_count = relevant_count + 1

    print("retrieved: " + str(retrieved_count) + " relevant: " + str(relevant_count) + " total relevant: " + str(image_amount*100))
    precision, recall = relevant_count / retrieved_count, relevant_count / (image_amount*100)
    print("Precision " + str(precision))
    print("Recall: " + str(recall))


# Uncomment this line to get a visual overview of the 20 most for an Image with the first argument being the
# category (0 - 99) and the second argument the index of the picture (1 - 100)
show_top_images(Image(2, 64))


# Uncomment this line to find the precision and recall of this method for a certain category
# (default: category = 1 (Bears), first_x_images = 20, image_amount = 5)
# (please ignore the "Corrupt JPEG data: premature end of data segment" warning, we're unsure where it comes from)
# calculate_precision_and_recall_for_category()
