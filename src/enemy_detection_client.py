#our vision detector model holds the image after processing it
#everytime an image is stream into the model, it will first compare the image with the image model is holding (held_img)
#if their normalised cross product is <= 0.9, then the image is considered different from the held_img
#This is to prevent the model from detecting the same image over and over again, which then improves the reactivity of the model
#as it can process stuff faster.


#server client code from : https://stackoverflow.com/questions/11352855/communication-between-two-computers-using-python-socket
import socket
import json
import pandas as pd

import re


RADAR_RANGE = (10,65,145,200)
SCREEN_WIDTH,SCREEN_HEIGHT = (1920, 1024)
class EnemyDetectorClient:
    def get_enemy_info():
        host='192.168.1.109' #client ip
        # host = '10.40.212.48'
        port = 6005
        
        server_port = 5000

        server = ('192.168.1.241', server_port)
        # server = ('10.40.36.226',server_port)
        
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(10)
        s.bind((host,port))
        data, addr = s.recvfrom(1024)
        data = data.decode('utf-8')
        data = re.sub(r"\'", "\"", str(data))
        data = re.sub(r"None", "\"null\"", data)
        
        data = json.loads(data)
        print("Received from server: " , data)
        if data['enemy_screen_coords'] == 'null':
            data['enemy_screen_coords'] = json.loads(data['enemy_screen_coords'])
        s.close()

        return data

if __name__ == '__main__':
    info = EnemyDetectorClient.get_enemy_info()
    print(info)
    
    