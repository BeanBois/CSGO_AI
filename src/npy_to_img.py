import numpy as np
import os
from input_data_utils import CSGOImageProcessor
import cv2
import pandas as pd

class NPYConvert:

    def __init__(self, npy_file):
        self.npy_file = npy_file
    
    def get_img(self):
        img = np.load(self.npy_file)
        return img[0]

    def update_image(self, npy_file):
        self.npy_file = npy_file



        
    
if __name__ == "__main__":
    # from ..src import input_data_utils
    
    
    #labelling positive images here, separate from the negative images
    image_files = os.listdir('./GameInterface/csgo_bomb_images/')
    csgo_image_processor = CSGOImageProcessor(None)
    converter = NPYConvert(None)
    for image_file in image_files:
        image_id = image_file[7:-4]
        print(image_id)
        print(image_file)
        converter.update_image('./GameInterface/csgo_bomb_images/' + image_file)
        image = converter.get_img()
        csgo_image_processor.update_image(image)
        # label = csgo_image_processor.get_label(image) #dict
        center_image = csgo_image_processor.get_center_image()

        center_image, label = csgo_image_processor.get_image_and_label(center_image)
        if label:
            dict =label 
            dict['image_id'] = image_id
            dict['label'] = 1 #1 for positive, 2 for negative
            df = pd.DataFrame(dict)# due to the way pd converts python dictionary to df,
                                   # stuff gets repeated on axis 0
            df.to_csv('./processed_images/positive_bblabel/proc_' + image_file[:-4] + '.csv', index=False)
            np.save('./processed_images/positive_images/proc_' + image_file[:-4] + '.npy', center_image)
            
        # proc_img = csgo_image_processor.label_img(center_image)
        # np.save(f'./processed_images/positive_images/proc_{image_file}.py',proc_img)
