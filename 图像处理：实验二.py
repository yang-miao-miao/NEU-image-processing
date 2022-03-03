import cv2
import numpy as np
from cv_function import *
plane=cv2.imread('./shiyan2/plane.bmp',0)
wirebond=cv2.imread('./shiyan2/Wirebond.tif',0)
coins=cv2.imread('./shiyan2/coins.png',0)
dowels=cv2.imread('./shiyan2/Dowels.tif',0)
rice=cv2.imread('./shiyan2/rice.png',0)
cov1=cv2.imread('./shiyan2/cov_1.jpg',0)
cov2=cv2.imread('./shiyan2/cov_2.jpg',0)
#1.1OTSU阈值分割
'''ret1,th1 = cv2.threshold(plane, 0, 255, cv2.THRESH_OTSU)  #方法选择为THRESH_OTSU
cv.imshow('OTSU',th1)#'''
#1.2迭代阈值分割
'''cv.imshow('th2',diedai(plane))'''
#1.3动态阈值分割
'''
#阈值取邻域平均值
th3=cv.adaptiveThreshold(plane,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,11,2)
#阈值取邻域加权平均值，呈高斯分布
th4=cv.adaptiveThreshold(plane,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,11,2)
cv.imshow('adapte_mean',th3)
cv.imshow('adapt_gaussian',th4)#'''
#2八方向边缘特征
'''for i,j in zip(['1','2','3','4','5','6','7','8'],[1,2,3,4,5,6,7,8]):
    cv.imshow(i, eight_canny(wirebond,j-1)) #理想低通'''
#3.1硬币检测
'''cv2.imshow('img2', circle_detect(coins))#'''
#3.2dowels检测
dowels_detect(dowels)
cv.imshow('img',dowels)#
#3.3米粒检测
'''rice_detect(rice)
cv.imshow('img',rice)#'''

cv2.waitKey(0)
cv2.destroyAllWindows()