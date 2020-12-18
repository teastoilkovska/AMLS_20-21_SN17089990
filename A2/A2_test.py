import numpy as np
from tensorflow.keras.models import load_model
from import_test_celeba import data
import matplotlib.pyplot as plt

model = load_model(r"C:\Users\Tea\PycharmProjects\classification\A2\smileclassification.h5")

IMG_SIZE = 64
n_channels = 1
X = np.array([i[0] for i in data]).reshape(-1, IMG_SIZE, IMG_SIZE, n_channels)
Y = [i[2] for i in data]



not_smile = 0
smile = 0
for i in Y:
    if i == 0:
        not_smile+=1
    else:
        smile+=1

print("NO:" +str(not_smile) + " MALES " +str(smile))

results = model.evaluate(X,Y)
print("test loss, test acc:", results)

import matplotlib.pyplot as plt

fig = plt.figure()
i = 0
j = 0
wrong_smile = 0
wrong_not_smile = 0
for img in X:
    #if i+1<100:
    #    y = fig.add_subplot(10, 10, i+1)
    orig = img
    data = img.reshape(1, 64, 64, n_channels)
    model_out = model.predict([data])
    #print("predicted" + str(model_out) + "Actuall" +str(Y[i-1]))
    model_out = np.argmax(model_out)

    if model_out == 0:
        str_label = "NO"
    else:
        str_label = "YES"
    if model_out != Y[i]:
        if Y[i] == 0:
            wrong_not_smile+=1
        else:
            wrong_smile+=1
        if (j+1)<20:
            j = j + 1
            y = fig.add_subplot(2, 10, j + 1)
            y.imshow(orig.reshape(64, 64, n_channels), cmap='gray')
            plt.title(str_label)
            y.axes.get_xaxis().set_visible(False)
            y.axes.get_yaxis().set_visible(False)

    i = i + 1
print("Wrong no " +str(wrong_not_smile ) +"wrong male" +str(wrong_smile))
plt.show()

