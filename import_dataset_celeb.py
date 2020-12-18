import cv2
import numpy as np
import os
from random import shuffle
from tqdm import tqdm
import csv


#MODIFY WITH THE DIRECTORY OF THE IMAGES AND THEIR CORRESPONDING LABELS
label_dir = r'Datasets\celeba\labels.csv'
img_dir = r'Datasets\celeba\img'

IMG_SIZE = 64

# In[4]:

file = open(label_dir)
reader = csv.reader(file)

celebrities = {}


for row in reader:
    split_string = str(row[0]).split("\t")
    celebrities[split_string[1]] = {'gender': split_string[2], 'smile': split_string[3]}


# In[6]:


def label_gender(img):
    if celebrities[img]['gender'] == '-1': return 0 #female
    elif celebrities[img]['gender'] == '1': return 1#male


def label_smile(img):
    if celebrities[img]['smile'] == '-1': return 0 #not smiling
    elif celebrities[img]['smile'] == '1': return 1#smile

def create_data():
    data = []

    for img in tqdm(os.listdir(img_dir)):
        label1 = label_gender(img)
        label2 = label_smile(img)
        path = os.path.join(img_dir, img)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = np.array(img)
        img = img / 255.0

        data.append([np.array(img), label1, label2])

    shuffle(data)
    np.save(r"dataceleb.npy", data)
    return data


if os.path.isfile(r"dataceleb.npy") is False:
    data = create_data()
else:
    data = np.load(r"dataceleb.npy", allow_pickle=True)