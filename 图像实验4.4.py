import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

muname = "./shiyan4/temp.bmp"
inname = 473

def cannymatch(img):   # 使用Canny算子进行边缘检测
    edge_output = cv2.Canny(img, 10, 300)
    cv2.namedWindow("Canny", 2)  # 创建一个窗口
    cv2.imshow('Canny', edge_output)  # 显示原始图片


def dajinmatch(img):   # 使用大津算子进行边缘检测
    x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(img, cv2.CV_16S, 0, 1)
    absX = cv2.convertScaleAbs(x)  # 转回uint8
    absY = cv2.convertScaleAbs(y)
    dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    # cv2.imshow("absX", absX)
    # cv2.imshow("absY", absY)
    cv2.imshow("dajin", dst)
    return dst


def addpath(name):  #方便图像序列增加
    cname = "./shiyan4/pic/" + str(name) + ".bmp"
    return cname

def erzhihua(grayim):   #二值化
    yuzhi = 40
    height,width = grayim.shape[:2]
    newgray = np.zeros((height, width), np.uint8)
    newgray = grayim // yuzhi * 255  #阈值分界二值
    return newgray

def xiangjian(dst,dstback):   #通过图像序列前后两帧进行相减突出重点
    n = 5 #中值滤波卷积核
    """ret, thresh1 = cv2.threshold(dst, yuzhi, 255, 0)  # 产生2值图
    ret, thresh2 = cv2.threshold(dstback, yuzhi, 255, 0)  # 产生2值图"""
    thresh1 = erzhihua(dst)
    thresh2 = erzhihua(dstback)
    h,w = img.shape[:2]
    jian = np.zeros((h,w))
    jian = abs(thresh2 - thresh1)
    jian = np.uint8(jian)
    jian1 = cv2.medianBlur(jian, n)  # 中值滤波,有效解决噪点
    cv2.imshow("jian", jian1)
    cv2.waitKey(100)
    return jian1

def kuangxuan(imgray,img):   #区域框选
    changex = 6
    changey = 6  #放大边框x,y
    yuzhi = 100
    ret, thresh = cv2.threshold(imgray, yuzhi, 255, 0)  # 产生2值图
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image = img.copy()
    # print(contours)
    #cv2.imshow("erzhitu", thresh)
    img = cv2.drawContours(img, contours, -1, (0, 255, 0), 5)  # img为三通道才能显示轮廓
    #cv2.imshow("drawCon",img)  #drawCon后的图像
    xmin = ymin = 10 ** 8
    xmax = ymax = 0
    for i in range(0, len(contours)):    #确定框选坐标
        x, y, w, h = cv2.boundingRect(contours[i])
        if x<xmin: xmin = x
        if y<ymin: ymin = y
        if x+w > xmax: xmax = x+w
        if y+h>ymax: ymax = y+h
    if xmin > 0:  # 防止边界溢出等情况
        cv2.rectangle(image, (xmin,ymin), (xmax+changex,ymax+changey), (255,0,0), 2)  #框选
        #newimage = image[y - cutn:y + h + cutn, x - cutn:x + w + cutn]  # 先用y确定高，再用x确定宽
        cv2.imshow("result", image)


if __name__ == "__main__":
    img = cv2.imread(addpath(inname))  # 图片
    template = cv2.imread("car.jpg")
    dstback = np.zeros((img.shape[0],img.shape[1]))
    while inname<497:
        img = cv2.imread(addpath(inname))  # 图片
        template = cv2.imread(muname)  # 模板
        grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        graytemplate = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        #cannymatch(grayimg)
        dst = dajinmatch(grayimg)
        imgwhite = xiangjian(dst,dstback)
        kuangxuan(imgwhite,img)
        dstback = dst
        #matchSQDIFF(grayimg, graytemplate) #标准平方差匹配
        #matchCCORR(grayimg,graytemplate) #标准相关匹配
        #matchSIFT(grayimg, graytemplate)  #sift匹配
        inname = inname + 1
    cv2.waitKey(0)
