#using csgo-gsi-python github repo code here
#REPO LINK: 

#server client inspired code from : https://stackoverflow.com/questions/11352855/communication-between-two-computers-using-python-socket
import socket
import json
NAME_OF_AGENT = 'beebeepop'
class client:
    
    def get_info(key):
        # host='192.168.1.109' #client ip
        # host='10.40.35.107' #client ip
        
        #to run by itself
        # host = '127.0.0.1'
        # port = 4005
        # server_port = 4000
        # server = ('127.0.0.1', server_port)
        
        #running with separate laptop 
        server_port = 4000
        port = 4005
        server = ('192.168.1.70', server_port)
        host='192.168.1.109' #client ip
        # host='10.40.35.107' #client ip
        
        # server = ('10.40.35.107', 4000)
        
        find_player = False
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # s.settimeout(5)
        
        s.bind((host,port))
        if key == "player":
            find_player = True
            key = "allplayers"
        s.sendto(key.encode('utf-8'), server)
        data, addr = s.recvfrom(1024*4)
        data = data.decode('utf-8')
        if data == "done":
            return
        # print("Received from server: " + data)
        # print(f'key: {key} data: {data}')
        data = json.loads(data)
        if find_player:
            data = client.get_player(data)
        # print("Received from server: " + str(data))
        # print("Received from server: " + str(data))
        s.close()
        # if key == 'position' or key == 'forward':
        #     coords = data.split(',')
        #     return coords
        return data
    
    def get_player(all_player_data):
        for player in all_player_data:
            if all_player_data[player]['name'] == NAME_OF_AGENT:
                return all_player_data[player]
        