import threading as th
from game_interface_server import GameServer
from enemy_detector_server import EnemyDetectorServer
import socket
host = '192.168.1.241'

# host = '127.0.0.1' #server ip
port = 4000

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.bind((host, port))
print("Server Started")
    
# _, addr = s.recvfrom(1024)
client =('192.168.1.109', 4005)

game_server = GameServer()
while True:
    EnemyDetectorServer.start_enemy_detection_model(s, client)
    game_server.get_action(s,client)
# enemy_detector_server = EnemyDetectorServer()

# thread1 = th.Thread(target=game_server.start_server)
# thread2 = th.Thread(target=EnemyDetectorServer.start_enemy_detection_model)

# #let 2 thread share the socket!
# thread1.start()
# thread2.start()
