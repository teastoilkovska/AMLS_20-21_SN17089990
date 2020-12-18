import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

import cv2
import numpy as np
import os
from random import shuffle
import csv
import tensorflow as tf
from import_cartoon import data

np.random.shuffle(data)

IMG_SIZE = 64

train = data[:-2250]
test = data[-2250:]

X = np.array([i[0] for i in train]).reshape(-1, IMG_SIZE, IMG_SIZE,1)
Y = [i[1] for i in train]

validation_x = np.array([i[0] for i in test]).reshape(-1, IMG_SIZE, IMG_SIZE,1)
validation_y = [i[1] for i in test]

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', padding="same", input_shape=(IMG_SIZE, IMG_SIZE, 1)))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (5, 5), activation='relu', padding="same"))
model.add(layers.MaxPooling2D((2, 2)))


model.add(layers.Conv2D(32, (3, 3), activation='relu', padding="same", input_shape=(IMG_SIZE, IMG_SIZE, 1)))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (5, 5), activation='relu', padding="same"))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())
model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(5))

opt = tf.keras.optimizers.Adam(learning_rate=0.001, amsgrad=False)

model.compile(optimizer=opt,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(X, Y, epochs=15,
                    validation_data=(validation_x, validation_y), batch_size=500)


csv_columns = ['Loss','Validation Loss', 'Acc', 'Validation Acc']
csv_file = "Face_shape.csv"

with open(csv_file, 'w') as f:
    for key in history.history.keys():
        f.write("%s,%s\n"%(key,history.history[key]))

if os.path.isfile(r"model_face.h5") is False:
    model.save(r"model_face.h5")
else:
    os.remove(r"model_face.h5")
    model.save(r"E:\model_face\model_face.h5")

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()