import numpy as np
from import_test_celeba import data as test_celeb
from tensorflow.keras.models import load_model
from import_test_cartoon_color import data as test_cartoon_color
from import_test_cartoon_gray import data as test_cartoon_gray

import matplotlib.pyplot as plt

#####################EVALUATE A1 and A2####################
IMG_SIZE = 64
n_channels_gray  = 1

X = np.array([i[0] for i in test_celeb]).reshape(-1, IMG_SIZE, IMG_SIZE, n_channels_gray)
Y1 = [i[1] for i in test_celeb]
Y2 = [i[2] for i in test_celeb]

modelA1 = load_model(r"A1\genderclassification.h5")
modelA2 = load_model(r"A2\smileclassification.h5")

results1 = modelA1.evaluate(X,Y1)
results2 = modelA2.evaluate(X,Y2)

###########################EVALUATE B1 AND B2#######################################


modelB2 = load_model(r"B2\model_eye.h5")

X_B2 = np.array([i[0] for i in test_cartoon_color]).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
Y_B2 = [i[1] for i in test_cartoon_color]

modelB1 = load_model(r"B1\model_face.h5")

X_B1 = np.array([i[0] for i in test_cartoon_gray]).reshape(-1, IMG_SIZE, IMG_SIZE, n_channels_gray)
Y_B1 = [i[1] for i in test_cartoon_gray]


results3 = modelB1.evaluate(X_B1,Y_B1)
results4 = modelB2.evaluate(X_B2,Y_B2)


print("A1: Test loss, Test acc:", results1)
print("A2: Test loss, Test acc:", results2)
print("B1: Test loss, Test acc:", results3)
print("B2: Test loss, Test acc:", results4)

##################################################################