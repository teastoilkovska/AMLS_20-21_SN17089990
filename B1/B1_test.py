import numpy as np
from tensorflow.keras.models import load_model
from import_test_cartoon_gray import data
import matplotlib.pyplot as plt

model = load_model(r"E:\model_face\model_face.h5")
model.summary()
IMG_SIZE = 64
n_channels = 1
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
    orig = img
    data = img.reshape(1, 64, 64, n_channels)
    model_out = model.predict([data])
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

        if (j+1)<11:
            j = j + 1
            y = fig.add_subplot(1, 12, j + 1)
            y.imshow(orig.reshape(64, 64, n_channels), cmap='gray')
            plt.title(str_label)
            y.axes.get_xaxis().set_visible(False)
            y.axes.get_yaxis().set_visible(False)

    i = i + 1

plt.show()

print("shape 0 " +str(shape0 ) +"shape 1 " +str(shape1))
print("shape 2 " +str(shape2 ) +"shape 3 " +str(shape2))
print("shape 4 " +str(shape4))