#using csgo-gsi-python github repo code here
#REPO LINK: 

#server client code from : https://stackoverflow.com/questions/11352855/communication-between-two-computers-using-python-socket
from csgo_gsi_python import GSI_SERVER_TRAINING
import socket
import re
from dict_to_unicode import asciify
import json
import pandas as pd
GSI_SERVER_TRAINING.start_server()
class server:
    def Main():
        # host = '192.168.0.12' #Server ip
        host = '192.168.1.109'
        # host = '127.0.0.1'
        port = 4000

        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.bind((host, port))

        print("Server Started")
        while True:
            #receive key
            data, addr = s.recvfrom(1024)
            
            if data is not None:
                data = data.decode('utf-8')
                print("Message from: " + str(addr))
                # print("From connected user: " + data)
                data = GSI_SERVER_TRAINING.get_info(data)
                data = re.sub(r"\'", "\"", str(data))
                data = re.sub(r"True", "\"1T\"", data)
                data = re.sub(r"False", "\"0F\"", data)
                data = re.sub(r"None", "\"null\"", data)
                # print("Sending: " + data)
                s.sendto(data.encode('utf-8'), addr)
        s.close()



class client:
    def get_info(key):
        host='192.168.1.241' #client ip
        port = 4005

        server = ('192.168.1.109', 4000)
        
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.bind((host,port))
        s.sendto(key.encode('utf-8'), server)
        data, addr = s.recvfrom(1024*4)
        data = data.decode('utf-8')
        # print("Received from server: " + data)
        data = json.loads(data)
        # print("Received from server: " + str(data))
        
        s.close()
        # if key == 'position' or key == 'forward':
        #     coords = data.split(',')
        #     return coords
        return data

if __name__=='__main__':
    server.Main()