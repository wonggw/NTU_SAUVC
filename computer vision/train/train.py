import numpy as np 
import netpart
import data_reader
import model as M
import tensorflow as tf 
import cv2 

import time

import os 
#if not os.path.exists('./model/'):
#	os.mkdir('./model/')

reader = data_reader.reader(height=720,width=1280,scale_range=[0.8,1.2])

def draw(img,c,b,multip,name):
	c = c[0]
	b = b[0]
	row,col,_ = b.shape
	# print(b.shape,c.shape)
	# print(row,col)
	for i in range(row):
		for j in range(col):
			# print(i,j)
			if c[i][j][0]>=-0.05:
				#print (c.max())
				x = int((b[i][j][0]+j+1/2)*multip)
				y = int((b[i][j][1]+i+1/2)*multip)
				w = int(b[i][j][2]*640)
				h = int(b[i][j][3]*480)
				cv2.rectangle(img,(x-w//2,y-h//2),(x+w//2,y+h//2),(0,255,0),2)
	cv2.imshow(name,img)
	cv2.waitKey(1)

def draw2(img,c,b,multip,name):
	c = c[0]
	b = b[0]
	row,col,_ = b.shape
	c = c.reshape([-1])
	ind = c.argsort()[-5:][::-1]
	for aaa in ind:
		# print(aaa)
		i = aaa//col
		j = aaa%col 
		x = int(b[i][j][0])+j*multip+multip//2
		y = int(b[i][j][1])+i*multip+multip//2
		w = int(b[i][j][2])
		h = int(b[i][j][3])
		cv2.rectangle(img,(x-w//2,y-h//2),(x+w//2,y+h//2),(0,255,0),2)
	cv2.imshow(name,img)
	cv2.waitKey(1)


i = 0
with tf.Session() as sess:
	saver = tf.train.Saver()
	writer = tf.summary.FileWriter('./log', sess.graph)	# write to file
	merge_op = tf.summary.merge_all()			# operation to merge all summary
	M.loadSess('./model/',sess,init=True)
	while True:
		i+=1
		img, train_dic = reader.get_img()
		#summary,loss_b1,loss_b2,loss_c,loss_cat,_,b,c = sess.run([merge_op,netpart.bias1_loss,
		loss_b1,loss_b2,loss_c,loss_cat,loss_pixel,loss_center,_,b,c = sess.run([netpart.bias1_loss,
			netpart.bias2_loss,netpart.conf_loss,netpart.cat_loss,
			netpart.pixelconf_loss,netpart.centerbias_loss,
			netpart.step,netpart.bias,netpart.conf],
			feed_dict={netpart.inpholder:[img],
			netpart.b_labholder:[train_dic[1]],
			netpart.c_labholder:[train_dic[0]],
			netpart.cat_labholder:[train_dic[2]],
			netpart.pixelconf_labelholder:[train_dic[3]],
			netpart.centerbias_labelholder:[train_dic[4]]})
		#writer.add_summary(i,summary)
		# Lastly, in your terminal or CMD, type this :
		# $ tensorboard --logdir /home/kyubey/Desktop/fyp/hand_gesture/train/log
		# open you google chrome, type the link shown on your terminal or CMD. (something like this: http://localhost:6006)
		if i%1==0:
			draw(img,c,b,32,'output')
			print('Iter:\t%d\tBias1_L:%.6f\tBias2_L:%.6f\tConf_L:%.6f\tCat_L:%.6f\tPixel_L:%.6f\tCenter_L:%.6f'%(i,loss_b1,loss_b2,loss_c,loss_cat,loss_pixel,loss_center))


		if i%1000==0 and i>0:
			saver.save(sess,'./model/MSRPN_%d.ckpt'%i)
