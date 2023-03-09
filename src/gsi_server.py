#using csgo-gsi-python github repo code here
#This file is mainly a test for gsi_server.py
# import csgo_gsi_python

#https://stackoverflow.com/questions/50612710/read-streaming-data-over-pipe-in-python
from socket import *
bufsize = 1024 # Modify to suit your needs
targetHost = "192.1.1.2"
listenPort = 8788

def forward(data, port):
print "Forwarding: '%s' from port %s" % (data, port)
sock = socket(AF_INET, SOCK_DGRAM)
sock.bind(("localhost", port)) # Bind to the port data came in on
sock.sendto(data, (targetHost, listenPort))

def listen(host, port):
listenSocket = socket(AF_INET, SOCK_DGRAM)
listenSocket.bind((host, port))
while True:
    data, addr = listenSocket.recvfrom(bufsize)
    forward(data, addr[1]) # data and port

listen("localhost", listenPort)

#stream data from laptop to laptop
from csgo_gsi_python import GSI_SERVER

GSI_SERVER.start_server()

print('done')