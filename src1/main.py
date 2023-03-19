import threading as th
from game_interface_server import GameServer
from enemy_detector_server import EnemyDetectorServer


game_server = GameServer()
# enemy_detector_server = EnemyDetectorServer()

thread1 = th.Thread(target=game_server.start_server)
thread2 = th.Thread(target=EnemyDetectorServer.start_enemy_detection_model)

thread1.start()
thread2.start()
