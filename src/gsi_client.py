#using csgo-gsi-python github repo code here
#REPO LINK: 

#server client inspired code from : https://stackoverflow.com/questions/11352855/communication-between-two-computers-using-python-socket
import socket
import select
import json
NAME_OF_AGENT = 'beebeepop'
class client:
    
    def get_info(key):

        
        #running with separate laptop 
        server_port = 4000
        port = 4005
        server = ('XXX.XXX.XXX.XXX', server_port) #ip address of the laptop communicating with the computer running this code
        host='XXX.XXX.XXX.XXX' #ip address of the laptop running the server
        
    
        find_player = False
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # s.settimeout(5)
        
        s.bind((host,port))
        if key == "player":
            find_player = True
            key = "allplayers"
        s.sendto(key.encode('utf-8'), server)
        ready = select.select([s], [], [], 0.5)
        while(not ready[0]):
            s.sendto(key.encode('utf-8'), server)
            ready = select.select([s], [], [], 0.5)
        if ready[0]:
            data, addr = s.recvfrom(1024*5)
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
        
if __name__ == '__main__':
    data = client.get_info('player')
    print(data)