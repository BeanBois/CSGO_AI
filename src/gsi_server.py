#using csgo-gsi-python github repo code here
#This file is mainly a test for gsi_server.py
# import csgo_gsi_python

#server client code from : https://stackoverflow.com/questions/11352855/communication-between-two-computers-using-python-socket
from csgo_gsi_python import GSI_SERVER_TRAINING
import socket
import re
from dict_to_unicode import asciify
import json
import pandas as pd
import socket
import json
GSI_SERVER_TRAINING.start_server()
class server:
    def Main():
        # host = '192.168.0.12' #Server ip
        host = '192.168.1.109'
        # host = '127.0.0.1'
        port = 4000

        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.bind((host, port))

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
    get_info('map') #test
    # start = time.time()
    # d = {'round_wins': None, 'current_spectators': 0, 'mode': 'casual', 'name': 'de_dust2', 'num_matches_to_win_series': 0, 'phase': 'live', 'round': 0, 'souvenirs_total': 0, 'team_ct': {'score': 0, 'consecutive_round_losses': 0, 'timeouts_remaining': 1, 'matches_won_this_series': 0}, 'team_t': {'score': 0, 'consecutive_round_losses': 0, 'timeouts_remaining': 1, 'matches_won_this_series': 0}}
    # d_encoded =  re.sub("None", "null", str(d))
    # d_l = re.sub(r"\'", "\"", d_encoded)
    # print(d_l)
    # d_encoded = d_l.encode('utf-8')
    # d_decoded = d_encoded.decode('utf-8')
    # print(type(d_decoded))
    # d_decode = json.loads(d_decoded)
    # print(d_decode==d)
    # print(f"Time taken: {time.time()-start} seconds")