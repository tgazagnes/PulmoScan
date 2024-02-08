import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from progiter import ProgIter
import cv2

import os, pathlib
import shutil #pour copier des fichiers vers les dossiers de preprocessing

l_label = ["Normal", "Viral_Pneumonia","Lung_Opacity", "COVID"]
l_diag = [0,1,2,3]
l_path=['Data/Sample/Normal/','Data/Sample/Viral_Pneumonia/','Data/Sample/Lung_Opacity/','Data/Sample/COVID/']

@st.cache_data
def init_raw_dataset():


  col_name = {'FILE NAME' : 'File',
            'FORMAT' : 'Format',
            'SIZE' : 'Size',
            }
  l_count = []
  l_df = []
        
  for index, diag in zip(range(0,4), l_diag):
    l_df.append(pd.read_excel('Data/Sample/'+ l_label[index]+ ".metadata.xlsx"))   
    l_count.append(l_df[index].shape[0])   
    l_df[index] = l_df[index].rename(col_name, axis=1)         
    l_df[index]['Diagnostic'] = diag                           
    l_df[index]['Path_images'] = l_path[index]

  return l_df

def any_random_file(l_df):
  selected_class = np.random.choice(range(4))
  file = np.random.choice(l_df[selected_class].File)
  return l_path[selected_class] + 'images/'+ file +'.png'

def load(file):
  return cv2.imread(file, cv2.IMREAD_GRAYSCALE)

def load_and_resize(file):
  img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
  return cv2.resize(img, (256, 256))

def load_mask(file):
  return cv2.imread(file.replace("images","masks"), cv2.IMREAD_GRAYSCALE)

def apply_mask(file):
  return cv2.bitwise_and(load_and_resize(file), load_mask(file))
    
def any_random_file(l_df):
  selected_class = np.random.choice(range(4))
  file = np.random.choice(l_df[selected_class].File)
  return l_path[selected_class] + 'images/'+ file+'.png'

def get_distrib(image: np.ndarray = None):
 
    image_a = np.round(image.astype(int) * 255.0 / np.max(image.astype(int)),0)  # NORMALISATION

    image_l = image_a.tolist()
    image_fl = []
    image_flze = []

    for i in range(0,len(image_l)):  # transfromation de l'array en liste de d'intensité de 0 à 255
        image_fl += image_l[i]

    for i in np.flatnonzero(image_fl): # liste d'intensité de 1 à 255
        image_flze.append(image_fl[i])
        
    intensite = [image_flze.count(i) for i in range(0,256)] # liste representant la distribution d'intensité non normalisé de l'image
    return image_flze, intensite
