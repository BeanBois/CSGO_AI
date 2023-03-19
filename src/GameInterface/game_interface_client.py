
import socket

class GameClient:

    def send_action(action):
        port = 5000
        host = '192.168.1.109'
        server = ('192.168.1.109', 5000)
        
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.bind((host, port))
        data = {'action' : action}
        data = str(data)
        s.sendto(data.encode('utf-8'), server)
        s.close()
        data, addr = s.recvfrom(1024)

