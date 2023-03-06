import numpy as np
import os
from input_data_utils import CSGOImageProcessor
import cv2

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
    
    

    image_files = os.listdir('./GameInterface/csgo_bomb_images')
    csgo_image_processor = CSGOImageProcessor(None)
    converter = NPYConvert(None)
    for image_file in image_files:
        converter.update_image('./GameInterface/csgo_bomb_images/' + image_file)
        image = converter.get_img()
        csgo_image_processor.update_image(image)
        center_image = csgo_image_processor.get_center_image()
        label = csgo_image_processor.get_label(image)
        if label:
            f = open('./processed_images/positive_bblabel/proc_' + image_file[:-4] + '.txt', 'w')
            f.write(f"0 {label[0]} {label[1]} {label[2]} {label[3]}")
            f.close()
        # proc_img = csgo_image_processor.label_img(center_image)
        # np.save(f'./processed_images/positive_images/proc_{image_file}.py',proc_img)
