import numpy as np 
import netpart
import model as M
import tensorflow as tf 
import cv2 

import time

def get_img_coord(img,c,b,cat,multip):
	# get the coordinations by c and b
	# multip is the gridsize.
	res_bias = []
	res_conf = []
	c = c[0]
	b = b[0]
	row,col,_ = b.shape
	# print(b.shape,c.shape)
	# print(row,col)
	for i in range(row):
		for j in range(col):
			# print(i,j)
			if c[i][j][0]>=0.0:
				x = abs(int((b[i][j][0]+j+1/2)*multip))
				y = abs(int((b[i][j][1]+i+1/2)*multip))
				w = abs(int(b[i][j][2]*640))
				h = abs(int(b[i][j][3]*480))
				cat1=cat[0][i][j]
				cat_ind= np.argmax(cat1)
				res_bias.append([x,y,w,h,cat_ind])
				res_conf.append(c[i][j][0])
	return res_bias,res_conf


def get_iou(inp1,inp2):
	x1,y1,w1,h1 = inp1[0],inp1[1],inp1[2],inp1[3]
	x2,y2,w2,h2 = inp2[0],inp2[1],inp2[2],inp2[3]
	#print y1,y2,h1,h2
	xo = min(abs(x1+w1/2-x2+w2/2), abs(x1-w1/2-x2-w2/2))
	yo = min(abs(y1+h1/2-y2+h2/2), abs(y1-h1/2-y2-h2/2))
	if abs(x1-x2) > (w1+w2)/2 or abs(y1-y2) > (h1+h2)/2:
		return 0
	if abs(float((x1-x2)*2)) < abs(w1-w2):
		xo = min(w1, w2)
	if abs(float((y1-y2)*2)) < abs(h1-h2):
		yo = min(h1, h2)
	overlap = xo*yo
	total = w1*h1+w2*h2-overlap
	#print 'ovlp',overlap
	#print 'ttl',total
	return float(overlap)/(total+0.00000000001)

def non_max_sup(coords,scr):
	# recursively get the max score in open list and delete the overlapped areas which is more than threshold
	non_max_thresh = 0.05
	open_coords = list(coords)
	open_scr = list(scr)
	result_coords = []
	
	while len(open_scr)>0:
		max_ind = np.argmax(np.array(open_scr))
		max_coord = open_coords[max_ind]
		result_coords.append(max_coord)
		del open_coords[max_ind]
		del open_scr[max_ind]
		#print len(open_scr)
		for i in range(len(open_scr),0,-1):
			iou = get_iou(open_coords[i-1],max_coord)
			#print iou
			if iou>non_max_thresh:
				del open_coords[i-1]
				del open_scr[i-1]
	return result_coords

def draw(img,coords):
	buff_img = img.copy()
	coords = list(coords)
	if coords:
		for i in range(len(coords)):
			x,y,w,h,cat = coords[i]
			cv2.rectangle(buff_img,(x-w//2,y-h//2),(x+w//2,y+h//2),(0,255,0),2)
			cv2.putText(buff_img, str(cat), (x,y), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 3, (0,0,255),6)
	cv2.imshow('result',buff_img)
	cv2.waitKey(1)

i = 0

sess = tf.Session()
M.loadSess('./model/',sess,init=True)

def get_coord_from_detection(img):

	b,c,cat,pc,cb = sess.run([netpart.bias,netpart.conf,netpart.cat,netpart.pixelconf,netpart.centerbias],feed_dict={netpart.inpholder:[img]})
	res_bias,res_conf=get_img_coord(img,c,b,cat,32)
	cb=np.clip(cb, 0, 1)
	cb=cb * 255
	cb = np.resize(cb, (23, 40))
	cb = cv2.resize(cb, (0,0), fx=10, fy=10) 
	cb = np.resize(cb, (1,230, 400,1))
	cb = cb.astype(np.uint8)
	print (cb.shape)
	cv2.imshow('abc',cb[0])
	cv2.waitKey(1)
	if not res_conf:
		result_coords=[]
	elif len(res_conf)==1:
		result_coords=res_bias
	else:
		result_coords=non_max_sup(res_bias,res_conf)
	draw(img,result_coords)
	return result_coords
#	return 1
