import model as M 
import numpy as np 
import tensorflow as tf 
import time 

def build_model(inp_holder):
	with tf.variable_scope('MASKRPN_V0'):
		inp_holder = tf.image.random_saturation(inp_holder,lower=0.5,upper=1.5)
		inp_holder = tf.image.random_contrast(inp_holder,lower=0.5,upper=2.0)
		inp_holder = tf.image.random_brightness(inp_holder,50)
		mod = M.Model(inp_holder)
		mod.dwconvLayer(3,5,stride=2,activation=M.PARAM_ELU,batch_norm=True)#640_ 2x2
		mod.convLayer(1,16,activation=M.PARAM_ELU,batch_norm=True)#640_ 2x2
		mod.dropout(0.8)
		mod.maxpoolLayer(3,2)#320_ 4x4
		mod.convLayer(3,18,stride=2,activation=M.PARAM_ELU,batch_norm=True)#160_ 8x8
		mod.dropout(0.8)
		mod.maxpoolLayer(3,2)#80_ 16x16
		mod.dwconvLayer(3,4,stride=2,activation=M.PARAM_ELU,batch_norm=True)#40_32x32
		l0=mod.convLayer(1,72,activation=M.PARAM_ELU,batch_norm=True)#640_ 2x2
		l1=mod.dwconvLayer(3,2,activation=M.PARAM_ELU,batch_norm=True)
		l2=mod.concat_to_current(l0)
		l3=mod.NIN(3,256,256,activation=M.PARAM_ELU)
		l4=mod.concat_to_current(l2)
		l3=mod.NIN(3,384,384,activation=M.PARAM_ELU)
		feature = mod.convLayer(1,8)

	return feature

inpholder = tf.placeholder(tf.float32,[None,None,None,3])
b_labholder = tf.placeholder(tf.float32,[None,None,None,4])
c_labholder = tf.placeholder(tf.float32,[None,None,None,1])
cat_labholder = tf.placeholder(tf.float32,[None,None,None,1])
pixelconf_labelholder = tf.placeholder(tf.float32,[None,None,None,1])
centerbias_labelholder = tf.placeholder(tf.float32,[None,None,None,1])

feature = build_model(inpholder)
bias,conf,cat,pixelconf,centerbias =tf.split(feature,[4,1,1,1,1], 3)
bias1,bias2=tf.split(bias,[2,2],3)
b1_labholder,b2_labholder=tf.split(b_labholder,[2,2],3)

with tf.variable_scope('bias_loss_x_y'):
	bias1_loss = tf.reduce_sum(tf.reduce_mean(tf.square(bias1 - b1_labholder)*c_labholder,axis=0)) #x,y
with tf.variable_scope('bias_loss_w_h'):
	bias2_loss = tf.reduce_sum(tf.reduce_mean(tf.square(tf.sqrt(tf.abs(bias2)) - tf.sqrt(tf.abs(b2_labholder)))*c_labholder,axis=0))#w,h

conf_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=c_labholder,logits=conf,name="Propose_bounding_box"))

cat_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=cat_labholder,logits=cat,name="Categorize_class"))


pixelconf_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=pixelconf_labelholder,logits=pixelconf,name="pixel_conf"))

with tf.variable_scope('centerbias_loss'):
	centerbias_loss = tf.reduce_sum(tf.reduce_mean(tf.square(centerbias - centerbias_labelholder)*pixelconf_labelholder,axis=0)) #x,y

with tf.variable_scope('Total_losses'):
	total_loss = (bias1_loss+ bias2_loss)+ centerbias_loss + conf_loss + cat_loss + pixelconf_loss
step = tf.train.RMSPropOptimizer(0.0001).minimize(total_loss)
