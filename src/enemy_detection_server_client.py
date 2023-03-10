#our vision detector model holds the image after processing it
#everytime an image is stream into the model, it will first compare the image with the image model is holding (held_img)
#if their normalised cross product is <= 0.9, then the image is considered different from the held_img
#This is to prevent the model from detecting the same image over and over again, which then improves the reactivity of the model
#as it can process stuff faster.
#code taken from repo: 
import numpy as np
from Models.Visual.Yolov5ForCSGO.aim_csgo.cs_model import load_model
from Models.Visual.Yolov5ForCSGO.utils.general import non_max_suppression, scale_coords, xyxy2xywh
from Models.Visual.Yolov5ForCSGO.utils.augmentations import letterbox
from Models.Visual.Yolov5ForCSGO.grabscreen import grab_screen
import torch
import cv2


###WINDOWS
import win32gui
import win32con




class EnemyRadarDetector:
    
    def __init__(self):
        pass
    pass


class EnemyScreenDetector:
    
    def __init__(self):
        self.CUDA_INFO = 'YES' if torch.cuda.is_available() else 'No'
        print('CUDA available : ' + self.CUDA_INFO)
        
        # Select GPU or CPU
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.half = device != 'cpu'
        self.imgsz = 640
        self.conf_thres = 0.8  # Confidence
        self.iou_thres = 0.05  # NMS IoU threshold   
        
        # Screen resolution
        self.x, self.y = (1920, 1080)
        self.re_x, self.re_y = (1920, 1080)
        
        self.model = load_model()
        self.stride = int(self.model.stride.max())
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.held_image = None
        self.enemy_screen_coords = {}
    
    
    #returns the x,y coordinates of the enemy
    # return (None,None) if no enemy detected
    def scan_for_enemy(self, image):
        if self.held_image is None:
            self.held_image = image
        else:
            if np.dot(image, self.held_image) <= 0.9:
                self.held_image = image
                return self._scan_for_enemy(image)
            else:
                return self.enemy_screen_coords

    def _scan_for_enemy(self, image):
        image_processed = self._process_image(image)
        self.held_image = image #update the held image since this is a new scan
        pred = self.model(image, augment=False, visualize=False)[0]
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, agnostic=False)
        aims = []
        for i, det in enumerate(pred):
            s = ''
            s += '%gx%g' % image_processed.shape[2:]
            gn = torch.tensor(image.shape)[[1, 0, 1, 0]]
            if len(det):
                det[:, :4] = scale_coords(image_processed.shape[2:], det[:, :4], image.shape).round()

                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)},"

                for *xyxy, conf, cls in reversed(det):
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                    line = (cls, *xywh)
                    aim = ('%g ' * len(line)).rstrip() % line  # str
                    aim = aim.split(' ')  # list
                    aims.append(aim)
                
                
                
                
                def _get_coords(x,y):
                    return self.x * float(x), self.y * float(y)
                
                #theres only 1 enemy player in the game (apart from the agent). As such this is possible
                
                if len(aims) == 0:
                    self.enemy_screen_coords = {
                        'body' : (None, None),
                        'head' : (None, None)
                    }
                #usually 
                elif len(aims) == 1:
                    x,y = _get_coords(aims[0][1], aims[0][2])
                    self.enemy_screen_coords = {
                        'body' : (x,y),
                        'head' : (None, None)
                    }
                else:
                    body_x,body_y = _get_coords(aims[1][1], aims[1][2])
                    head_x,head_y = _get_coords(aims[0][1], aims[0][2])
                    
                    self.enemy_screen_coords = {
                        'body' : (body_x, body_y),
                        'head' : (head_x, head_y)
                    }
        return self.enemy_screen_coords

    def _process_image(self,img0):
        img0 = cv2.resize(img0, (self.re_x, self.re_y))

        img = letterbox(img0, self.imgsz, stride=self.stride)[0]

        img = img.transpose((2, 0, 1))[::-1]
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()
        img /= 255.
        if len(img.shape) == 3:
            img = img[None]
        return img


ENEMY_SCREEN_DETECTOR = EnemyScreenDetector()
        


# #server client code from : https://stackoverflow.com/questions/11352855/communication-between-two-computers-using-python-socket
# import socket
# import re
# from dict_to_unicode import asciify
# import json
# import pandas as pd
# from Models.Visual.Yolov5ForCSGO import Yolov5ForCSGO
# class server:
#     def start_object_detection_model():
#         host = '127.0.0.1' #server ip
#         port = 4000

#         s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
#         s.bind((host, port))

#         print("Server Started")
#         while True:
#             #receive key
#             data, addr = s.recvfrom(1024)
            
#             if data is not None:
#                 data = data.decode('utf-8')
#                 print("Message from: " + str(addr))
#                 # print("From connected user: " + data)
#                 data = GSI_SERVER_TRAINING.get_info(data)
#                 data = re.sub(r"\'", "\"", str(data))
#                 data = re.sub(r"True", "\"1T\"", data)
#                 data = re.sub(r"False", "\"0F\"", data)
#                 data = re.sub(r"None", "\"null\"", data)
#                 # print("Sending: " + data)
#                 s.sendto(data.encode('utf-8'), addr)
#         # s.close()



# class client:
#     def get_info(key):
#         host='192.168.1.241' #client ip
#         port = 4005

#         server = ('192.168.1.109', 4000)
        
#         s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
#         s.bind((host,port))
#         s.sendto(key.encode('utf-8'), server)
#         data, addr = s.recvfrom(1024*4)
#         data = data.decode('utf-8')
#         # print("Received from server: " + data)
#         data = json.loads(data)
#         # print("Received from server: " + str(data))
        
#         s.close()
#         # if key == 'position' or key == 'forward':
#         #     coords = data.split(',')
#         #     return coords
#         return data
