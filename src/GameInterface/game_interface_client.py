
import socket

class GameClient:

    def send_action(action, done=False):
        port = 4005
        host = '192.168.1.109'
        server = ('192.168.1.241', 4000)
        
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.bind((host, port))
        data = {'action' : action, 'done' : "1" if done else ""}
        data = str(data)
        s.sendto(data.encode('utf-8'), server)
        s.close()
        # data, addr = s.recvfrom(1024)

