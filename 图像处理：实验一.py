import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from cv_function import *
food=cv.imread('./shiyan1/Food.jpg',0)
circuit=cv.imread('./shiyan1/Circuit_original.tif',0)
city=cv.imread('./shiyan1/city.jpg')
height = food.shape[0]
width = food.shape[1]
gs1=gaosi_noise(circuit,0,0.01)
gs2=gaosi_noise(circuit,0,0.02)
gs3=gaosi_noise(circuit,0,0.03)
jy1=jiaoyan_noise(circuit,0.01)
jy2=jiaoyan_noise(circuit,0.02)
jy3=jiaoyan_noise(circuit,0.03)
#第一问
'''
# 线性变换
linear = np.zeros((height, width), np.uint8)
linear = linear_transform(food,1.5,linear)
cv.imshow('linear',linear)
#对数变换
log = log_transform(42,food)
cv.imshow('log',log)
#均衡化
equ=equalize(food)
cv.imshow('equ',equ)
cv.imshow('origion',food)'''
#第二问

'''for i,j in zip(['1','2','3','4','5','6'],[gs1,gs2,gs3,jy1,jy2,jy3]):
    #cv.imshow(i,cv.GaussianBlur(j,(3,3),0)) #3*3高斯滤波
    #cv.imshow(i,cv.GaussianBlur(j,(5,5),0))#5*5高斯滤波
    #cv.imshow(i,cv.GaussianBlur(j,(7,7),0))#7*7高斯滤波
    #cv.imshow(i,cv.medianBlur(j,3))#3*3中值滤波
    #cv.imshow(i,cv.medianBlur(j,5))#5*5中值滤波
    cv.imshow(i,cv.medianBlur(j,7))#7*7中值滤波'''
#第三问
'''for i,j in zip(['1','2','3'],[30,60,90]):
    cv.imshow(i, lixiangditong(circuit,j)) #理想低通'''
'''for i,j in zip([1,1,1,2,2,2],[30,60,90,30,60,90]):
    plt.figure()
    plt.title(('n=',i,'D=',j))
    plt.imshow(batewosi(circuit,j,i), cmap='gray')
    plt.axis('off')  
plt.show()    #巴特沃斯滤波器'''
'''for i,j in zip(['1','2','3'],[30,60,90]):
    cv.imshow(i, gaosi(circuit,j)) #高斯低通'''
#第四问
stronger= deHaze(city / 255.0)
cv.imshow('city',city)
cv.imshow('stronger',stronger)#'''
cv.waitKey(0)
cv.destroyAllWindows()