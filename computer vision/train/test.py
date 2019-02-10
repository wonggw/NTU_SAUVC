import gesture_detection
import time
from camera_module import camera_thread

camera_thread = camera_thread()
camera_thread.start()

while True:
	t1=time.time()
	img = camera_thread.read()
	coords = gesture_detection.get_coord_from_detection(img)
	#print (coords)
	t2=time.time()
	print (t2-t1)
