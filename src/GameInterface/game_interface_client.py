
import socket

class GameClient:

    def send_action(action, done=False):
        port = 5005
        host = '192.168.1.109'
        server = ('192.168.1.241', 5000)
        
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.bind((host, port))
        data = {'action' : action, 'done' : "1" if done else ""}
        data = str(data)
        s.sendto(data.encode('utf-8'), server)
        data = s.recv(1024)
        data = data.decode('utf-8')
        while data != "done":
            data = s.recv(1024)
            data = data.decode('utf-8')
            print(data)
        s.close()
        # data, addr = s.recvfrom(1024)

