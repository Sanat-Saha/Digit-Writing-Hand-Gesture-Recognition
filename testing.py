import socket, traceback
from xml.etree import cElementTree as ET
import re
import sys, os

host = "192.168.137.1"
port = 5555

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
s.bind((host, port))

i=0

while True:
	try:
		message, address = s.recvfrom(8192)
		print(message)
		print(i)
		i += 1
	except KeyboardInterrupt:
		break

# while True:
# 	print "Hi"
# 	try:
# 		message, address = s.recvfrom(8192)
# 		if message:
# 			print(message)
# 		else:
# 			print("Hi")
# 		# print(message)
# 	except:
# 		print("Hi")
# 		break