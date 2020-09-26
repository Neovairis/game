# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 21:11:41 2018

@author: DELL
"""
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn import preprocessing
import pandas as pd

seq1=open("D:/separatedata/datachallenge_seq1.txt")
lines=seq1.readlines()
cycles=[]
sensor=np.empty((21,len(lines))).tolist()

for line in lines:
    a=line.strip().split(" ")
    cycles.append(float(a[1]))
    for i in range(21):
        sensor[i].append(float(a[i+5]))
        del sensor[i][0] 

for i in range(21):
    min_max_scaler=preprocessing.MinMaxScaler(feature_range=(-1,1))
    sensor[i]=min_max_scaler.fit_transform(np.array(sensor[i]))
#sentrans=np.array(sensor).transpose()

y=[]
for i in range(len(cycles)):
    if (len(cycles)-cycles[i])>=130:
        y.append(130)
    else:
        y.append(len(cycles)-cycles[i])
y=np.array(y)

#print(sentrans,y)
"""
fig1=plt.figure(1)
plt.plot(cycles,sensor[1],label='signal2')
plt.plot(cycles,sensor[2],label='signal3')
plt.xlabel('cycles')
plt.ylabel('signal')
plt.legend()
plt.grid()
"""
del sensor[0]
del sensor[3:5]
del sensor[6]
del sensor[11]
del sensor[12:14]
#以上读取数据并完成特征选择及标准化
s=[] #s是timewindow之后的时间特征，将其喂入cnn
for i in range(14):
    s.append(pd.Series(sensor[i]).rolling(window=30).mean())
"""做出时间窗和目标曲线图
fig2=plt.figure(2)
for i in range(14):
    plt.plot(cycles,s[i])

plt.xlabel('cycles')
plt.ylabel('signal')
plt.grid()

fig3=plt.figure(3)
plt.plot(cycles,y)
"""
s=np.array(s).transpose()#s是(223,14)的
#print(s.shape)


sess=tf.InteractiveSession()
xs=tf.placeholder(tf.float32,shape=[None,14])
ys=tf.placeholder(tf.float32,shape=[None,1])
xs_reshape=tf.reshape(xs,[-1,223,14,1])
epoch=250
for i in range(epoch):#设置学习率
    if i<=200:
        learning_rate=0.001
    else:
        learning_rate=0.0001
#开始设置卷积层
def weight_variable(shape):
    #initial=tf.glorot_normal_initializer(shape,tf.float32)#Xavier初始化
    initial=tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)
def bias_variable(shape):
    initial=tf.constant(0.1,shape=shape)
    return tf.Variable(initial)
def conv2d(xs,w):
    return tf.nn.conv2d(xs,w,strides=[1,1,1,1],padding='SAME')
#4个完全相同的卷积层进行特征提取
w_conv1=weight_variable([10,1,1,10])#[10,1,1,10]10*1的卷积核，1个颜色通道，10个卷积核
b_conv1=bias_variable([10])
h_conv1=tf.nn.tanh(conv2d(xs_reshape,w_conv1)+b_conv1)

w_conv2=weight_variable([10,1,10,10])
b_conv2=bias_variable([10])
h_conv2=tf.nn.tanh(conv2d(h_conv1,w_conv2)+b_conv2)

w_conv3=weight_variable([10,1,10,10])
b_conv3=bias_variable([10])
h_conv3=tf.nn.tanh(conv2d(h_conv2,w_conv3)+b_conv3)

w_conv4=weight_variable([10,1,10,10])
b_conv4=bias_variable([10])
h_conv4=tf.nn.tanh(conv2d(h_conv3,w_conv4)+b_conv4)
#1个卷积层汇总提取的特征
w_conv5=weight_variable([3,1,10,1])
b_conv5=bias_variable([1])
h_conv5=tf.nn.tanh(conv2d(h_conv4,w_conv5)+b_conv5)
#flatten
h_conv5=tf.reshape(h_conv5,[-1,223*14*1])
keep_prob=tf.placeholder(tf.float32)
h_conv5_drop=tf.nn.dropout(h_conv5,keep_prob)
#全连接层
w_fcl=weight_variable([223*14*1,100])
b_fcl=bias_variable([100])
h_fcl=tf.nn.tanh(tf.matmul(h_conv5_drop,w_fcl)+b_fcl)
#输出层
w_o=weight_variable([100,1])
b_o=bias_variable([1])
h_o=tf.matmul(h_fcl,w_o)+b_o

cost=tf.losses.mean_squared_error(labels=ys,predictions=h_o)
train=tf.train.AdamOptimizer(learning_rate).minimize(cost)



tf.global_variables_initializer().run()
a=plt.figure(1,figsize=(6,5))
plt.ion()
for i in range(epoch):
    #batch=len(s)
    #if i%50==0:
        train.run(feed_dict={xs:s.reshape(-1,14),ys:y.reshape(-1,1),keep_prob:0.5})
        output=sess.run(h_o,feed_dict={xs:s.reshape(-1,14),ys:y.reshape(-1,1),keep_prob:1.0})
        """
        a.clear()
        plt.plot(cycles,y,'r',label='target')
        plt.legend()
        #plt.plot(cycles,output,'b',label='regression')
        plt.legend()
        plt.grid()
        plt.show()
        plt.pause(0.1)
        """
        
print(output)
plt.ioff()