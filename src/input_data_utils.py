import numpy as np
import cv2
import pandas as pd
from  Models.Visual.Yolov5ForCSGO.aim_csgo.cs_model import load_model
from Models.Visual.Yolov5ForCSGO.utils.augmentations import letterbox
from Models.Visual.Yolov5ForCSGO.utils.general import non_max_suppression, scale_coords, xyxy2xywh
import torch
import os
import matplotlib.pyplot as plt


#lets take a functional approach with processing images
#basically we will have a function that is kinda like lambda, where we take an image and return 
# the cropped out image center and the cropped out image radar


class CSGOImageProcessor:
    RADAR_IMAGE_DIMENSION = None
    CENTER_CROP_DIMENSION = [0, 0, 1920, 1080]

    conf_thres = 0.8  # Confidence
    iou_thres = 0.05  # NMS IoU threshold

    # Screen resolution
    x, y = (1920, 1080)
    re_x, re_y = (1920, 1080)
    imgsz = 640
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    half = device != 'cpu'
    def __init__(self, image):
        self.image = image
        
    def update_image(self, image):
        self.image = image

    def get_radar_image(self):
        # get the radar image
        # radar_image = self.image[RADAR_IMAGE_DIMENSION]
        radar_image = self.image
        return radar_image
    

    def get_center_image(self):
        # get the center image
        
        # center_image = self.image[CENTER_CROP_DIMENSION]
        center_image = self.image
        center_image = cv2.resize(center_image, (CSGOImageProcessor.re_x, CSGOImageProcessor.re_y))
        return center_image

    def visualize_scan_center_image_for_enemy(self, center_image):
        # scan the center image for an enemy
        model = load_model()
        stride = int(model.stride.max())
        names = model.module.names if hasattr(model, 'module') else model.names
        

        img = letterbox(center_image, CSGOImageProcessor.imgsz, stride=stride)[0]

        img = img.transpose((2, 0, 1))[::-1]
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(CSGOImageProcessor.device)
        img = img.half() if CSGOImageProcessor.half else img.float()
        img /= 255.
        if len(img.shape) == 3:
            img = img[None]
            
        pred = model(img, augment=False,visualize=False)[0]
        pred = non_max_suppression(pred, CSGOImageProcessor.conf_thres, CSGOImageProcessor.iou_thres, agnostic=False)
        aims = []
        for i, det in enumerate(pred):
            s = ''
            s += '%gx%g' % img.shape[2:]
            gn = torch.tensor(center_image.shape)[[1, 0, 1, 0]]
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], center_image.shape).round()

                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)},"

                for *xyxy, conf, cls in reversed(det):
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                    line = (cls, *xywh)
                    aim = ('%g ' * len(line)).rstrip() % line  # str
                    aim = aim.split(' ')  # list
                    aims.append(aim)  # Aim at the target
            if len(aims):
                for i, det in enumerate(aims):
                    _, x_center, y_center, width, height = det
                    x_center, width = CSGOImageProcessor.re_x * float(x_center), CSGOImageProcessor.re_x * float(width)
                    y_center, height = CSGOImageProcessor.re_y * float(y_center), CSGOImageProcessor.re_y * float(height)
                    top_left = (int(x_center - width / 2.), int(y_center - height / 2.))
                    bottom_right = (int(x_center + width / 2.)), (int(y_center + height / 2.))
                    color = (0, 255, 0)  # Show targets with green boxes
                    cv2.rectangle(center_image, top_left, bottom_right, color, thickness=3)
                    
        cv2.namedWindow('csgo-detect', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('csgo-detect', CSGOImageProcessor.re_x // 3, CSGOImageProcessor.re_y // 3)
        cv2.imshow('csgo-detect', center_image)
        cv2.waitKey(0)
        # Press q to end the program
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
        return 0
    
    def label_img(self, center_image):
        # scan the center image for an enemy
        model = load_model()
        stride = int(model.stride.max())
        names = model.module.names if hasattr(model, 'module') else model.names
        

        img = letterbox(center_image, CSGOImageProcessor.imgsz, stride=stride)[0]

        img = img.transpose((2, 0, 1))[::-1]
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(CSGOImageProcessor.device)
        img = img.half() if CSGOImageProcessor.half else img.float()
        img /= 255.
        if len(img.shape) == 3:
            img = img[None]
            
        pred = model(img, augment=False,visualize=False)[0]
        pred = non_max_suppression(pred, CSGOImageProcessor.conf_thres, CSGOImageProcessor.iou_thres, agnostic=False)
        aims = []
        for i, det in enumerate(pred):
            s = ''
            s += '%gx%g' % img.shape[2:]
            gn = torch.tensor(center_image.shape)[[1, 0, 1, 0]]
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], center_image.shape).round()

                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)},"

                for *xyxy, conf, cls in reversed(det):
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                    line = (cls, *xywh)
                    aim = ('%g ' * len(line)).rstrip() % line  # str
                    aim = aim.split(' ')  # list
                    aims.append(aim)  # Aim at the target
            if len(aims):
                first = True
                for i, det in enumerate(aims):
                    if first:
                        first = False
                        continue
                    _, x_center, y_center, width, height = det
                    x_center, width = CSGOImageProcessor.re_x * float(x_center), CSGOImageProcessor.re_x * float(width)
                    y_center, height = CSGOImageProcessor.re_y * float(y_center), CSGOImageProcessor.re_y * float(height)
                    top_left = (int(x_center - width / 2.), int(y_center - height / 2.))
                    bottom_right = (int(x_center + width / 2.)), (int(y_center + height / 2.))
                    color = (0, 255, 0)  # Show targets with green boxes
                    cv2.rectangle(center_image, top_left, bottom_right, color, thickness=3)
                
        return center_image

    def get_label(self, center_image):

        #from https://stackoverflow.com/questions/56115874/how-to-convert-bounding-box-x1-y1-x2-y2-to-yolo-style-x-y-w-h
        def yolobbox2bbox(x,y,w,h):
            x1, y1 = x-w/2, y-h/2
            x2, y2 = x+w/2, y+h/2
            return x1, y1, x2, y2
        area = None
           
        # scan the center image for an enemy
        model = load_model()
        stride = int(model.stride.max())
        names = model.module.names if hasattr(model, 'module') else model.names
        

        img = letterbox(center_image, CSGOImageProcessor.imgsz, stride=stride)[0]

        img = img.transpose((2, 0, 1))[::-1]
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(CSGOImageProcessor.device)
        img = img.half() if CSGOImageProcessor.half else img.float()
        img /= 255.
        if len(img.shape) == 3:
            img = img[None]
            
        pred = model(img, augment=False,visualize=False)[0]
        pred = non_max_suppression(pred, CSGOImageProcessor.conf_thres, CSGOImageProcessor.iou_thres, agnostic=False)
        aims = []
        for i, det in enumerate(pred):
            s = ''
            s += '%gx%g' % img.shape[2:]
            gn = torch.tensor(center_image.shape)[[1, 0, 1, 0]]
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], center_image.shape).round()

                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)},"

                for *xyxy, conf, cls in reversed(det):
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                    line = (cls, *xywh)
                    aim = ('%g ' * len(line)).rstrip() % line  # str
                    aim = aim.split(' ')  # list
                    aims.append(aim)  # Aim at the target
            if len(aims):
                first = True
                for i, det in enumerate(aims):
                    if len(aims) > 1 and first:
                        first = False
                        continue
                    _, x_center, y_center, width, height = det
                    x_center, y_center, width, height = float(x_center), float(y_center), float(width), float(height)
                    area = width * height
                    return { 'bb' : yolobbox2bbox(x_center, y_center, width, height), 'area': area}
                    # color = (0, 255, 0)  # Show targets with green boxes
                    # cv2.rectangle(center_image, top_left, bottom_right, color, thickness=3)

        # return center_image
    
    def scan_center_image_for_enemy(self, center_image):
        # scan the center image for an enemy
        model = load_model()
        stride = int(model.stride.max())
        names = model.module.names if hasattr(model, 'module') else model.names
        

        img = letterbox(center_image, CSGOImageProcessor.imgsz, stride=stride)[0]

        img = img.transpose((2, 0, 1))[::-1]
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(CSGOImageProcessor.device)
        img = img.half() if CSGOImageProcessor.half else img.float()
        img /= 255.
        if len(img.shape) == 3:
            img = img[None]
            
        pred = model(img, augment=False,visualize=False)[0]
        pred = non_max_suppression(pred, CSGOImageProcessor.conf_thres, CSGOImageProcessor.iou_thres, agnostic=False)
        aims = []
        for i, det in enumerate(pred):
            s = ''
            s += '%gx%g' % img.shape[2:]
            gn = torch.tensor(center_image.shape)[[1, 0, 1, 0]]
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], center_image.shape).round()

                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)},"

                for *xyxy, conf, cls in reversed(det):
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                    line = (cls, *xywh)
                    aim = ('%g ' * len(line)).rstrip() % line  # str
                    aim = aim.split(' ')  # list
                    aims.append(aim)  # Aim at the target
        return aims,CSGOImageProcessor.x,CSGOImageProcessor.y

    def get_image_and_label(self,center_image):
                # scan the center image for an enemy
        model = load_model()
        stride = int(model.stride.max())
        names = model.module.names if hasattr(model, 'module') else model.names
        bottom_right = None
        top_left = None
        area = None
        img = letterbox(center_image, CSGOImageProcessor.imgsz, stride=stride)[0]

        img = img.transpose((2, 0, 1))[::-1]
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(CSGOImageProcessor.device)
        img = img.half() if CSGOImageProcessor.half else img.float()
        img /= 255.
        if len(img.shape) == 3:
            img = img[None]
            
        pred = model(img, augment=False,visualize=False)[0]
        pred = non_max_suppression(pred, CSGOImageProcessor.conf_thres, CSGOImageProcessor.iou_thres, agnostic=False)
        aims = []
        for i, det in enumerate(pred):
            s = ''
            s += '%gx%g' % img.shape[2:]
            gn = torch.tensor(center_image.shape)[[1, 0, 1, 0]]
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], center_image.shape).round()

                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)},"

                for *xyxy, conf, cls in reversed(det):
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                    line = (cls, *xywh)
                    aim = ('%g ' * len(line)).rstrip() % line  # str
                    aim = aim.split(' ')  # list
                    aims.append(aim)  # Aim at the target
            if len(aims):
                first = True
                for i, det in enumerate(aims):
                    if len(aims) > 1 and first:
                        first = False
                        continue
                    _, x_center, y_center, width, height = det
                    x_center, width = CSGOImageProcessor.re_x * float(x_center), CSGOImageProcessor.re_x * float(width)
                    y_center, height = CSGOImageProcessor.re_y * float(y_center), CSGOImageProcessor.re_y * float(height)
                    top_left = (int(x_center - width / 2.), int(y_center - height / 2.))
                    bottom_right = (int(x_center + width / 2.)), (int(y_center + height / 2.))
                    area = width * height
                    color = (0, 255, 0)  # Show targets with green boxes
                    cv2.rectangle(center_image, top_left, bottom_right, color, thickness=3)
                    return center_image,{ 'bb' : [top_left[0], top_left[1], bottom_right[0], bottom_right[1]], 'area': area}
        return None, None

from http.server import BaseHTTPRequestHandler, HTTPServer
import sys
import time
import json


class MyServer(HTTPServer):
    def __init__(self, server_address, token, RequestHandler):
        self.auth_token = token

        super(MyServer, self).__init__(server_address, RequestHandler)

        # You can store states over multiple requests in the server 
        self.round_phase = None


class MyRequestHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers['Content-Length'])
        body = self.rfile.read(length).decode('utf-8')

        self.parse_payload(json.loads(body))

        self.send_header('Content-type', 'text/html')
        self.send_response(200)
        self.end_headers()

    def is_payload_authentic(self, payload):
        if 'auth' in payload and 'token' in payload['auth']:
            return payload['auth']['token'] == server.auth_token
        else:
            return False

    def parse_payload(self, payload):
        # Ignore unauthenticated payloads
        if not self.is_payload_authentic(payload):
            return None

        round_phase = self.get_round_phase(payload)

        if round_phase != self.server.round_phase:
            self.server.round_phase = round_phase
            print('New round phase: %s' % round_phase)

    def get_round_phase(self, payload):
        if 'round' in payload and 'phase' in payload['round']:
            return payload['round']['phase']
        else:
            return None

    def log_message(self, format, *args):
        """
        Prevents requests from printing into the console
        """
        return





if __name__ == '__main__':  
    # lets test the class
    csgo_image_processor = CSGOImageProcessor(None)
    test_images = os.listdir('test_images')
    for test_image in test_images:
        image = cv2.imread('test_images/' + test_image)
        csgo_image_processor.update_image(image)
        radar_image = csgo_image_processor.get_radar_image()
        center_image = csgo_image_processor.get_center_image()
        # csgo_image_processor.visualize_scan_center_image_for_enemy(center_image)
        csgo_image_processor.get_label(center_image)
        
    # server = MyServer(('localhost', 3000), 'MYTOKENHERE', MyRequestHandler)

    # print(time.asctime(), '-', 'CS:GO GSI Quick Start server starting')

    # try:
    #     server.serve_forever()
    # except (KeyboardInterrupt, SystemExit):
    #     pass

    # server.server_close()
    # print(time.asctime(), '-', 'CS:GO GSI Quick Start server stopped')