import cv2 
import threading
import time 
import os


class camera_thread(threading.Thread):
	def __init__(self):
#		self.http = 'http://'
#		self.ip_address = '192.168.1.74'
#		self.url =  self.http + self.ip_address + ':4747/mjpegfeed?1024x768'
		self.cap = cv2.VideoCapture("/home/kyubey/Desktop/suavc/video/Underwater RED Green Poles.mp4")
		#self.cap.set(14, 0.01)  #exposure
		_,self.img = self.cap.read()
		#self.img = hist_equal(self.img)
		threading.Thread.__init__(self)

	def run(self):
		while True:
			time.sleep(0.03)
			try:
				_,self.img = self.cap.read()
				#self.img = hist_equal(self.img)
				#equ= cv2.cvtColor(self.img, cv2.COLOR_BGR2HLS)
				#equ[:,:,1] = cv2.equalizeHist(equ[:,:,1])
				#self.img = cv2.cvtColor(equ, cv2.COLOR_HLS2BGR)
			except:
				pass

	def read(self):
		return self.img 
