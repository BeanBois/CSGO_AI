import os
import numpy as np
import cv2

FILE_DIR = './GameInterface/csgo_bomb_images/'
# FILE_DIR_PROC = './processed_images/positive_images/'
FILE_DIR_PROC = './GameInterface/csgo_bomb_images'
from input_data_utils import CSGOImageProcessor


ip = CSGOImageProcessor(None)

for file in os.listdir(FILE_DIR_PROC):
    
    image = np.load(os.path.join(FILE_DIR_PROC, file))[0]
    ip.update_image(image)
    cv2.imshow('image', image)
    cv2.waitKey(0)
    