import cv2
import numpy as np
import os
from random import shuffle
from tqdm import tqdm
import csv



#MODIFY WITH YOUR DIRECTORY
img_dir = r'Datasets\cartoon_set\img'
label_dir = r'Datasets\cartoon_set\labels.csv'

IMG_SIZE = 64


file = open(label_dir)
reader = csv.reader(file)

cartoons = {}

# In[5]:


for row in reader:
    split_string = str(row[0]).split("\t")
    cartoons[split_string[3]] = {'eye_color': split_string[1], 'face_shape': split_string[2]}


# In[6]:
def label_eye(img):
    if cartoons[img]['eye_color'] == '0':
        return 0  # [0, 0, 0, 0, 0]
    elif cartoons[img]['eye_color'] == '1':
        return 1  # [0, 1, 0, 0, 0]
    elif cartoons[img]['eye_color'] == '2':
        return 2  # [0, 0, 1, 0, 0]
    elif cartoons[img]['eye_color'] == '3':
        return 3  # [0, 0, 0, 1, 0]
    elif cartoons[img]['eye_color'] == '4':
        return 4  # [0, 0, 0, 0, 1]




def create_data():
    data = []

    for img in tqdm(os.listdir(img_dir)):
        label = label_eye(img)
        path = os.path.join(img_dir, img)
        img = cv2.imread(path)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        data.append([np.array(img), label])

    shuffle(data)
    np.save(r"C:\Users\Tea\PycharmProjects\classification\colordatacartoon.npy", data)

    return data


# In[ ]:


# In[8]:

if os.path.isfile(r"colordatacartoon.npy") is False:
    data = create_data()
else:
    data = np.load(r"colordatacartoon.npy", allow_pickle=True)