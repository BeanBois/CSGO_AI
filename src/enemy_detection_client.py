#our vision detector model holds the image after processing it
#everytime an image is stream into the model, it will first compare the image with the image model is holding (held_img)
#if their normalised cross product is <= 0.9, then the image is considered different from the held_img
#This is to prevent the model from detecting the same image over and over again, which then improves the reactivity of the model
#as it can process stuff faster.


#server client code from : https://stackoverflow.com/questions/11352855/communication-between-two-computers-using-python-socket
import socket
import json
import pandas as pd




RADAR_RANGE = (10,65,145,200)
SCREEN_WIDTH,SCREEN_HEIGHT = (1920, 1024)
class EnemyDetectorClient:
    def get_enemy_info():
        host='192.168.1.109' #client ip
        port = 6005

        server = ('192.168.1.241', 6000)
        
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.bind((host,port))
        data, addr = s.recvfrom(1024)
        data = data.decode('utf-8')
        data = json.loads(data)
        print("Received from server: " + str(data))
        s.close()

        return data
