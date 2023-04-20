import threading as th
from game_interface_server import GameServer
from enemy_detector_server import EnemyDetectorServer
import socket
host = 'xxx.xxx.xxx.xxx' #ip of the computer that will be running the code


port = 5000

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.bind((host, port))
game_client = ('xxx.xxx.xxx.xxx', 5005) #ip address of the laptop communicating with the computer running this code
enemy_detector_client = ('xxx.xxx.xxx.xxx', 6005)#ip address of the laptop communicating with the computer running this code



game_server = GameServer()


while True:
    game_server.get_action(s, game_client)
    EnemyDetectorServer.enemy_detect(s, enemy_detector_client)


