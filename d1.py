# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 10:16:29 2018

@author: DELL
"""
#鸭哥是个大傻子
import matplotlib.pyplot as plt
import numpy as np

x=open("D:/unit1.txt",mode='r',encoding='utf-8')
lines=x.readlines()
list1=[]
list2=[]
list3=[]
list4=[]
list5=[]
list6=[]
list7=[]
list8=[]
list9=[]
list10=[]
for line in lines:
    a=line.split(" ")
    b=a[5:6]
    c=a[6:7]
    d=a[1:2]
  
    list1.append(b)
    list2.append(c)
    list3.append(d)
  
   
   
   

fig1=plt.figure()
plt.scatter(list1,list2)
plt.xlabel('signal1')
plt.ylabel('signal2')
plt.legend()


fig2=plt.figure()
plt.plot(list3,list1,color='red',label='signal1')
plt.plot(list3,list2,color='green',label='signal2')
plt.xlabel('time,cycles')
plt.ylabel('signal')
plt.legend()
plt.grid()


