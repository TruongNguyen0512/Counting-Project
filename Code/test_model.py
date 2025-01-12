import tensorflow as tf
import cv2
import numpy as np
import time
from matplotlib import pyplot as plt

def load_and_preprocess_image(image_path):
    # Đọc ảnh
    img = cv2.imread(image_path)
    # Resize về kích thước 64x64
    img = cv2.resize(img, (64, 64))
    # Normalize
    img = img / 255.0