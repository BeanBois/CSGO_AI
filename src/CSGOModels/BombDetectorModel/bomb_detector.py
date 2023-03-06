import torch
import os
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from PIL import Image

# This is a dataset wrapper for the CSGO bomb dataset
# Our dataset has the following structure:
    # the training images themselves <the training images, size H,W>
    # target -- a dictionary with
        # 'boxes' : the bounding box coordinates of the bomb in the image [x0 y0 x1 y1] (top left and bottom right corners)
            #this comes from the enemy detection model
            #format FloatTensor[N, 4] with values between 0 and H and 0 and W, N being # of bbox
        # 'labels': the class label for each bounding box (1 for bombDefusing) <(Int64Tensor[N])
        # 'area' : the area of the bounding box <Tensor[N]>
        # 'image_id' : will be the rand_seed number we used to gen images. <Int64Tensor[1]> used during evaluation
# images have the following names: Frame_<rand_seed>.npy
# labels have the following names: Frame_<rand_seed>.txt
# 
class BombDataset(torch.utils.data.Dataset):
    

class BombDetectionModel(nn.Module):
    def __init__(self):
        
    
    #for trasnform here we can utilise the CSGOImageProcessor class (as lambda) in addition to the ToTensor() function
    def __init__(self, root, transform=ToTensor()):
        self.root = root
        self.transform = transform
        self.images = list(sorted(os.listdir(os.path.join(root, "images")))) #Training Images
        self.labels = list(sorted(os.listdir(os.path.join(root, "labels")))) #Training Labels
        #we find the la
            
    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "images", self.images[idx])
        img = Image.open(img_path).convert("RGB")
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        if self.transform is not None:
            img = self.transform(img)
        return img, target
    
    def __len__(self):
        return len(self.images)