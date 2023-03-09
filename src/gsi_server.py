
























































import socket
import json
import re
import time
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