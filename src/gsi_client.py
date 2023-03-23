#using csgo-gsi-python github repo code here
#REPO LINK: 

#server client inspired code from : https://stackoverflow.com/questions/11352855/communication-between-two-computers-using-python-socket
import socket
import json

class client:
    
    def get_info(key):
        # host='192.168.1.109' #client ip
        # host='10.40.35.107' #client ip
        # host='10.40.35.107' #client ip
        
        #to run by itself
        host = '127.0.0.1'
        port = 4005
        server_port = 4000
        server = ('127.0.0.1', server_port)
        
        # server = ('192.168.1.70', 4000)
        # server = ('192.168.1.70', 4000)
        # server = ('10.40.35.107', 4000)
        
        
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.bind((host,port))
        s.sendto(key.encode('utf-8'), server)
        data, addr = s.recvfrom(1024*4)
        data = data.decode('utf-8')
        # print("Received from server: " + data)
        print(f'key: {key} data: {data}')
        data = json.loads(data)
        # print("Received from server: " + str(data))
        print("Received from server: " + str(data))
        s.close()
        # if key == 'position' or key == 'forward':
        #     coords = data.split(',')
        #     return coords
        return data
    