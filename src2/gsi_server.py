#using csgo-gsi-python github repo code here
#REPO LINK: 

#server client inspired code from : https://stackoverflow.com/questions/11352855/communication-between-two-computers-using-python-socket
from csgo_gsi_python import GSI_SERVER_TRAINING
import socket
import re
import json
import pandas as pd
from pynput.mouse import Controller, Button

RESET= 5000
class server:
    MouseController = Controller()
    def start_csgo_gsi_server():
        GSI_SERVER_TRAINING.start_server()

        host = 'xxx.xxx.xxx.xxx' #ip address of the laptop running the server

        port = 4000
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.bind((host, port))
        i = 0
        print("Server Started")
        while True:
            
            i +=1
            if i % RESET ==0:
                i = 0
                GSI_SERVER_TRAINING.shutdown()
                GSI_SERVER_TRAINING.start_server()
            
            #receive key
            data, addr = s.recvfrom(1024)
            
            

            if data is not None:
                data = data.decode('utf-8')
                print(data)
                print("Message from: " + str(addr))
                # print("From connected user: " + data)
                if data == "switch spectator target":
                    server.MouseController.click(Button.left, 1)
                    data = "done"
                    s.sendto(data.encode('utf-8'), addr)
                    continue
                data = GSI_SERVER_TRAINING.get_info(data)
                print(data)
                data = re.sub(r"\'", "\"", str(data))
                data = re.sub(r"True", "\"1T\"", data)
                data = re.sub(r"False", "\"0F\"", data)
                data = re.sub(r"None", "\"null\"", data)
                # print("Sending: " + data)
                s.sendto(data.encode('utf-8'), addr)


if __name__=='__main__':
    server.start_csgo_gsi_server()