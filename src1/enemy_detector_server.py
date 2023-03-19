from Visual.Yolov5ForCSGO.aim_csgo.cs_model import cs_model
from Visual.Yolov5ForCSGO.utils.general import general
from Visual.Yolov5ForCSGO.utils.augmentations import augmentations
from Visual.Yolov5ForCSGO.aim_csgo.grabscreen import grabscreen

#server client code from : https://stackoverflow.com/questions/11352855/communication-between-two-computers-using-python-socket
import socket
import pandas as pd
import torch


###WINDOWS
import win32gui
import win32con

import cv2
import numpy as np
from matplotlib import pyplot as plt


RADAR_RANGE = (10,65,145,200)
SCREEN_WIDTH,SCREEN_HEIGHT = (1920, 1080)
class EnemyRadarDetector:
    
    def __init__(self):
        pass
    def scan_for_enemy(self, image):

        #color definition
        red_lower = np.array([30,30,210])
        red_upper = np.array([60,60,255])

        mask = cv2.inRange(image, red_lower, red_upper)
        connectivity = 4  
        # Perform the operation
        output = cv2.connectedComponentsWithStats(mask, connectivity, cv2.CV_32S)
        # Get the results

        num_labels = output[0]-1

        centroids = output[3][1:]



        #print results
        print ('number of dots, should be 4:',num_labels)
        print ('array of dot center coordinates:',centroids)
        return num_labels > 0, centroids


class EnemyScreenDetector:
    
    def __init__(self):
        self.CUDA_INFO = 'YES' if torch.cuda.is_available() else 'No'
        print('CUDA available : ' + self.CUDA_INFO)
        
        # Select GPU or CPU
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.half = self.device != 'cpu'
        self.imgsz = 640
        self.conf_thres = 0.8  # Confidence
        self.iou_thres = 0.05  # NMS IoU threshold   
        
        # Screen resolution
        self.x, self.y = (1920, 1080)
        self.re_x, self.re_y = (1920, 1080)
        
        self.model = cs_model.load_model()
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
        pred = general.non_max_suppression(pred, self.conf_thres, self.iou_thres, agnostic=False)
        aims = []
        for i, det in enumerate(pred):
            s = ''
            s += '%gx%g' % image_processed.shape[2:]
            gn = torch.tensor(image.shape)[[1, 0, 1, 0]]
            if len(det):
                det[:, :4] = general.scale_coords(image_processed.shape[2:], det[:, :4], image.shape).round()

                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)},"

                for *xyxy, conf, cls in reversed(det):
                    xywh = (general.xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
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

        img = augmentations.letterbox(img0, self.imgsz, stride=self.stride)[0]

        img = img.transpose((2, 0, 1))[::-1]
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()
        img /= 255.
        if len(img.shape) == 3:
            img = img[None]
        return img


ENEMY_SCREEN_DETECTOR = EnemyScreenDetector()
ENEMY_RADAR_DETECTOR = EnemyRadarDetector()   





#This code will be ran by computer thats learning the game model. <The Reinfocement Learning Agent>
#This pc can be in any OS, But i will be using MACXOS
class EnemyDetectorServer:
    def start_enemy_detection_model():
        
        
        host = '127.0.0.1' #server ip
        port = 4000

        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.bind((host, port))
        print("Server Started")
        
        # _, addr = s.recvfrom(1024)
        client =('xxx', 4000)
        
        while True:
            #receive the coordinates of the enemy on screen, and if enemy is present
            img = grabscreen(region=(0, 0, 1920, 1080)) #TODO: decide on region
            x0 = RADAR_RANGE[0]
            y0 = RADAR_RANGE[1]
            x1 = RADAR_RANGE[2]
            y1 = RADAR_RANGE[3]
            radar_img = img[y0:y1, x0:x1]
            cv2.imshow('radar', radar_img)
            cv2.waitKey(1) #comment out, jsut to check implementation
            enemy_on_radar = ENEMY_RADAR_DETECTOR.scan_for_enemy(radar_img)
            if enemy_on_radar:
                enemy_screen_coords = ENEMY_SCREEN_DETECTOR.scan_for_enemy(img)
                if enemy_screen_coords is not None:
                    data = {"enemy_on_screen" : "1", "enemy_screen_coords" : enemy_screen_coords}
                else:
                    data = {"enemy_on_screen" : "1", "enemy_screen_coords" : "null"}           
            else:
                data = {"enemy_on_screen" : "null", "enemy_screen_coords" : "null"}
            #then process the data from client, specifically
            #see if the enemy is present, and if so, get the coordinates of the enemy
            s.sendto(data.encode('utf-8'), client)
        # s.close()