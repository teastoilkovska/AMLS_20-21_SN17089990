import cv2
import numpy as np
import os
from random import shuffle
from tqdm import tqdm
import csv

#MOFIDY WITH YOUR DIRECTORY
label_dir = r'Datasets\cartoon_set_test\labels.csv'
img_dir = r'Datasets\cartoon_set_test\img'
IMG_SIZE = 64


file = open(label_dir)
reader = csv.reader(file)

cartoons = {}

# In[5]:


for row in reader:
    split_string = str(row[0]).split("\t")
    cartoons[split_string[3]] = {'eye_color': split_string[1], 'face_shape': split_string[2]}


# In[6]:
def label_face(img):
    if cartoons[img]['face_shape'] == '0':
        return 0  # [0, 0, 0, 0, 0]
    elif cartoons[img]['face_shape'] == '1':
        return 1  # [0, 1, 0, 0, 0]
    elif cartoons[img]['face_shape'] == '2':
        return 2  # [0, 0, 1, 0, 0]
    elif cartoons[img]['face_shape'] == '3':
        return 3  # [0, 0, 0, 1, 0]
    elif cartoons[img]['face_shape'] == '4':
        return 4  # [0, 0, 0, 0, 1]

def create_data():
    data = []

    for img in tqdm(os.listdir(img_dir)):
        label = label_face(img)
        path = os.path.join(img_dir, img)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img / 255.0

        data.append([np.array(img), label])

    shuffle(data)
    np.save(r"graytestcartoon.npy", data)

    return data


# In[ ]:


# In[8]:

if os.path.isfile(r"graytestcartoon.npy") is False:
    data = create_data()
else:
    data = np.load(r"graytestcartoon.npy", allow_pickle=True)