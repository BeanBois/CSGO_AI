
import socket

class GameClient:

    def send_action(action):
        port = 5000
        host = '192.168.1.109'
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.bind((host, port))
        data = {'action' : action}
        s.sendto(data.encode('utf-8'), addr)
        s.close()
        data, addr = s.recvfrom(1024)

