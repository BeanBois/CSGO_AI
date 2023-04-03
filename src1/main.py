import threading as th
from game_interface_server import GameServer
from enemy_detector_server import EnemyDetectorServer
import socket
host = '192.168.1.241'

# host = '127.0.0.1' #server ip
port = 5000

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.bind((host, port))
s.settimeout(5)
# s.setblocking(0)    
game_client = ('192.168.1.109', 5005)
enemy_detector_client = ('192.168.1.109', 6005)
# host = '127.0.0.1' #server ip
# port = 4000

# s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
# s.bind((host, port))
# print("Server Started")
    
# # _, addr = s.recvfrom(1024)
# client =('192.168.1.109', 4005)

game_server = GameServer()
# enemy_detector_server = EnemyDetectorServer()

while True:
    try:
        game_server.get_action(s, game_client)
    except:
        pass
    try:
        EnemyDetectorServer.enemy_detect(s, enemy_detector_client)
    except:
        pass
    # enemy_detector_server.start_enemy_detection_model(s, client)
# thread1 = th.Thread(target=game_server.get_action, args=(s, client)
# thread2 = th.Thread(target=EnemyDetectorServer.start_enemy_detection_model, args=(s, client)

#let 2 thread share the socket!
# thread1.start()
# thread2.start()
