The way to run the demos changed a bit since the report was delivered last monday.

To run the deep learning demo see the README file in the DeepLearning folder.

To run traditional_demo.py, deep_learning_demo.py and fusion_demo.py:

1. Install the sklearn, pywavelets, opencv-python, matplotlib and numpy packages
2. Run classify.py in the Traditional folder and rename the result .pkl file to random_forest_tr.pkl
2. Run classify_random_forest.py in the DeepLearning folder and rename the result .pkl file to random_forest_dl.pkl
3. Drag these files into the MMSR folder
4. Run any of the *_demo.py files

Make sure you are using 64 bit python otherwise you will run into memory errors.

We have added 3 images that are not in the Corel10k dataset> To find related images for these images, add 0_4.npy, 0_55.npy and 0_61.npy to db_features in DeepLearning and use either:
Image(4, -400)
Image(55, -5500)
Image(61, -6100)
