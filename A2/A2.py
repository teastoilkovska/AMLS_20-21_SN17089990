import cv2
import numpy as np
import os
from random import shuffle
from tqdm import tqdm
import csv
import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from import_dataset_celeb import data
from sklearn.model_selection import KFold

np.random.shuffle(data)

n_folds = 5
acc_per_fold = []
loss_per_fold = []
highest_accuracy = 0
kfold = KFold(n_splits=n_folds, shuffle=True)
fold_no = 1

IMG_SIZE = 64
n_channels = 1
X = np.array([i[0] for i in data]).reshape(-1, IMG_SIZE, IMG_SIZE, n_channels)
Y = np.array([i[2] for i in data])


for train, test in kfold.split(X, Y):
    model = models.Sequential()
    model.add(layers.Conv2D(14, (5, 5), activation='relu', padding="same", input_shape=(IMG_SIZE, IMG_SIZE, n_channels)))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(10, (5, 5), activation='relu', padding="same"))
    model.add(layers.MaxPooling2D((2, 2)))
    #model.summary()

    model.add(layers.Flatten())
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(2))

    opt = tf.keras.optimizers.Adam(learning_rate=0.001, amsgrad=False)

    model.compile(optimizer=opt,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience = 1)
    #history = model.fit(X, Y, epochs=20,
    #                validation_data=(validation_x, validation_y), batch_size=100, callbacks=[es])

    history = model.fit(X[train], Y[train], epochs=20, validation_data=(X[test], Y[test]), batch_size=100, callbacks=[es])

    print('------------------------------------------------------------------------')
    print(f'Training for fold {fold_no} ...')

    print("Score for "+str(fold_no)+"is "+str( history.history['val_acc'][-1]))

    acc_per_fold.append(history.history['val_acc'][-1])
    loss_per_fold.append(history.history['val_loss'][-1])

    # Increase fold number
    fold_no = fold_no + 1

    import csv
    if history.history['val_acc'][-1] > highest_accuracy:

        highest_accuracy = history.history['val_acc'][-1]

        csv_columns = ['Loss','Validation Loss', 'Acc', 'Validation Acc']
        csv_file = "Smile_classification.csv"

        with open(csv_file, 'w') as f:
            for key in history.history.keys():
                f.write("%s,%s\n"%(key,history.history[key]))

        if os.path.isfile(r"smileclassification.h5") is False:
            model.save(r"smileclassification.h5")
        else:
            os.remove(r"smileclassification.h5")
            model.save(r"smileclassification.h5")


print("Highest accuracy is" + str(highest_accuracy))
print("Accuracies are" + str(acc_per_fold))