import win32api
import win32con
from PIL import ImageGrab
import numpy as np
import cv2

def get_screen():
    width = win32api.GetSystemMetrics(win32con.SM_CXVIRTUALSCREEN)
    height = win32api.GetSystemMetrics(win32con.SM_CYVIRTUALSCREEN)
    dimensions = (0, 0, width, height)
    img = ImageGrab.grab(dimensions)
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return np.array(img)