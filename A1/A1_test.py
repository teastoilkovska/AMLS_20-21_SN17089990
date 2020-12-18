import numpy as np
from tensorflow.keras.models import load_model
from import_test_celeba import data
model = load_model(r"C:\Users\Tea\PycharmProjects\classification\A1\genderclassification.h5")

model.summary()


IMG_SIZE = 64
n_channels = 1
X = np.array([i[0] for i in data]).reshape(-1, IMG_SIZE, IMG_SIZE, n_channels)
Y = [i[1] for i in data]

n_f = 0
n_m = 0

for i in Y:
    if i == 0:
        n_f+=1
    else:
        n_m+=1

print("FEMALES:" +str(n_f) + " MALES " +str(n_m))

results = model.evaluate(X,Y)
print("test loss, test acc:", results)

import matplotlib.pyplot as plt

fig = plt.figure()
i = 0
j = 0
wrong_f = 0
wrong_m = 0
for img in X:
    orig = img
    data = img.reshape(1, 64, 64, n_channels)
    model_out = model.predict([data])
    model_out = np.argmax(model_out)

    if model_out == 0:
        str_label = "Female"
    else:
        str_label = "Male"
    if model_out != Y[i]:
        if Y[i] == 0:
            wrong_f+=1
        else:
            wrong_m+=1
        if (j+1)<20:
            j = j + 1
            y = fig.add_subplot(2, 10, j + 1)
            y.imshow(orig.reshape(64, 64, n_channels), cmap='gray')
            plt.title(str_label)
            y.axes.get_xaxis().set_visible(False)
            y.axes.get_yaxis().set_visible(False)

    i = i + 1

plt.show()

print("Wrong female " +str(wrong_f ) +"wrong male" +str(wrong_m))