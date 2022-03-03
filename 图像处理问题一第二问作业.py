import numpy as np
import cv2 as cv
#高斯滤波
'''dst = cv2.GaussianBlur(src,ksize,sigmaX,sigmay,borderType)
        src: 输入图像矩阵,可为单通道或多通道，多通道时分别对每个通道进行卷积
        dst:输出图像矩阵,大小和数据类型都与src相同
        ksize:高斯卷积核的大小，宽，高都为奇数，且可以不相同
        sigmaX: 一维水平方向高斯卷积核的标准差
        sigmaY: 一维垂直方向高斯卷积核的标准差，默认值为0，表示与sigmaX相同
        borderType:填充边界类型'''
#双边滤波
'''dst = cv2.bilateralFilter(src,d,sigmaColor,sigmaSpace,borderType)
        src: 输入图像对象矩阵,可以为单通道或多通道
        d:用来计算卷积核的领域直径，如果d<=0，从sigmaSpace计算d
        sigmaColor：颜色空间滤波器标准偏差值，决定多少差值之内的像素会被计算（构建灰度值模板）
        sigmaSpace:坐标空间中滤波器标准偏差值。如果d>0，设置不起作用，否则根据它来计算d值（构建距离权重模板）'''
#引导滤波
'''导向滤波
    cv2.ximgproc.guidedFilter(guide,src,radius,eps,dDepth)
        guide: 导向图片，单通道或三通道
        src: 输入图像对象矩阵,可以为单通道或多通道
        radius:用来计算卷积核的领域直径
        eps:规范化参数， eps的平方类似于双边滤波中的sigmaColor（颜色空间滤波器标准偏差值）
       (regularization term of Guided Filter. eps2 is similar to the sigma 
         in the color space into bilateralFilter.)
        dDepth: 输出图片的数据深度'''
img=cv.imread('./images/1/california_22_13.bmp')
img1=cv.GaussianBlur(img,(3,3),1)
img2=cv.bilateralFilter(img,9,75,75)
img3=cv.ximgproc.guidedFilter(img,img,3,2,-1)#test_partten选择1，其余3
'''cv.imwrite('E:/Desktop/imgwork/learn/que2/coins/img1.png',img1)
cv.imwrite('E:/Desktop/imgwork/learn/que2/coins/img2.png',img2)
cv.imwrite('E:/Desktop/imgwork/learn/que2/coins/img3.png',img3)'''
cv.imshow('img',img)
cv.imshow('a',img1)
cv.imshow('b',img2)
cv.imshow('c',img3)
cv.waitKey(0)
cv.destroyAllWindows()