import numpy as np
import pandas as pd
import cv2 # OpenCV for image processing functions like LBP
import os # For file system operations
from tqdm import tqdm # For visualizing loop progress
import glob

from sklearn.model_selection import train_test_split # For splitting data
from sklearn.neighbors import KNeighborsClassifier # KNN Classifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from skimage.feature import local_binary_pattern

from google.colab import drive
drive.mount('/content/drive/')

DATA_PATHS = {
    'jaffe_train': '/content/drive/MyDrive/prjct/JAFFE_dataset/train',
    'jaffe_test': '/content/drive/MyDrive/prjct/JAFFE_dataset/test',
    'ck_train': '/content/drive/MyDrive/prjct/CK_dataset/train',
    'ck_test': '/content/drive/MyDrive/prjct/CK_dataset/test'
}

# Image parameters
TARGET_SIZE = (48, 48)
LBP_POINTS = 8
LBP_RADIUS = 1
