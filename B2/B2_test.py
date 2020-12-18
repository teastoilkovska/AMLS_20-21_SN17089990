import numpy as np
from tensorflow.keras.models import load_model
from import_test_cartoon_color import data
import matplotlib.pyplot as plt
import cv2

model = load_model(r"E:\eye_model\model_eye.h5")

IMG_SIZE = 64
n_channels = 3
X = np.array([i[0] for i in data]).reshape(-1, IMG_SIZE, IMG_SIZE, n_channels)
Y = [i[1] for i in data]


results = model.evaluate(X,Y)
print("test loss, test acc:", results)


fig = plt.figure()
i = 0
j = 0
shape0 = 0
shape1= 0
shape2 = 0
shape3 = 0
shape4 = 0

for img in X:
    #if i+1<100:
    #    y = fig.add_subplot(10, 10, i+1)
    orig = img
    data = img.reshape(1, 64, 64, n_channels)
    model_out = model.predict([data])
    #print("predicted" + str(model_out) + "Actuall" +str(Y[i-1]))
    model_out = np.argmax(model_out)

    str_label =  str(model_out) + "/" + str(Y[i])
    if model_out != Y[i]:
        if Y[i] == 0:
            shape0+=1
        elif Y[i] == 1:
            shape1+=1
        elif Y[i] == 2:
            shape2+=1
        elif Y[i] ==3:
            shape3+=1
        elif Y[i] ==4:
            shape4+=1

        if (j+1)<40:
            j = j + 1
            y = fig.add_subplot(4, 10, j + 1)

            y.imshow(cv2.cvtColor(orig.reshape(64, 64, n_channels), cv2.COLOR_BGR2RGB))
            plt.title(str_label)
            y.axes.get_xaxis().set_visible(False)
            y.axes.get_yaxis().set_visible(False)

    i = i + 1

plt.show()

print("color 0 " +str(shape0 ) +"color 1 " +str(shape1))
print("color 2 " +str(shape2 ) +"color 3 " +str(shape2))
print("color 4 " +str(shape4))