
import socket
import time
import json
class GameClient:

    def send_action(action, done=False):
        port = 5005
        host = '192.168.1.109'
        server = ('192.168.1.241', 5000)
        # buffer = (host, port)
        
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.bind((host, port))
        # action = re.sub(r"None", "\"null\"", action)
        data = {'action' : action, 'done' : "1" if done else ""}
        # data = str(data)
        data = json.dumps(data)
        s.sendto(data.encode('utf-8'), server)
        data = s.recv(1024)
        data = data.decode('utf-8')
        while data != "done":
            # #send old data to back to server
            # #honesty unsure if this will even work
            # #issue 1: can you send data to your own port? probs yes
            # #issue 2 : what if we create a infinite loop of sending data to ourselves?
            # s.sendto(data.encode('utf-8'), buffer)
            # time.sleep(0.01)
            
            data = s.recv(1024)
            data = data.decode('utf-8')
            print(data)
        s.close()
        # data, addr = s.recvfrom(1024)

if __name__ == "__main__":
    action = [0,0,0,0,0,0,0,0,0,0,0,None,None]
    # action[6] = 1
    action[8] = 1
    action = [str(x) for x in action]
    action = ','.join(action)
    for i in range(1920):
        GameClient.send_action(action,False)