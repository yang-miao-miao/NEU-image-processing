import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import math

muname = "./shiyan4/temp.bmp"
inname = 473  #第一帧
Z = 15  #前十帧作为模板
step = 4  # 设置加速度求取步长进行滤波
global savelo   #位置保存数组 savelo[i] = [lx,ly,rx,ry]左下lx，ly；右上rx，ry 坐标
global start
global avlx,avly,avrx,avry,delx,dely,derx,dery

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

def push(xmin,ymin,xmax,ymax,inname):   #将Z之前的方框信息导入
    savelo[inname-start - 1] = np.array([xmin,ymin,xmax,ymax])
    print(inname - start -1)
    print(savelo[inname - start - 1])  #观察框是怎么变的

def kuangxuan(imgray,img,inname,Z):   #区域框选
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
        if inname <Z:
            push(xmin,ymin,xmax+changex,ymax+changey,inname)  #把这次的坐标导入push中给savelo
        #newimage = image[y - cutn:y + h + cutn, x - cutn:x + w + cutn]  # 先用y确定高，再用x确定宽
        cv2.imshow("result", image)

def yucemean(img,inname):  #通过之前保留的savelo预测轨迹，平均增长方式
    #print("hai")
    #print(inname,start)
    savelo[inname - start - 1] = [0,0,0,0]
    yulx = savelo[inname - start - 2][0] + avlx
    yuly = savelo[inname - start - 2][1] + avly
    yurx = savelo[inname - start - 2][2] + avrx
    yury = savelo[inname - start - 2][3] + avry
    savelo[inname - start-1] = [yulx,yuly,yurx,yury]
    cv2.rectangle(img, (yulx, yuly), (yurx,yury), (0,255,255), 2)  # 框选
    # newimage = image[y - cutn:y + h + cutn, x - cutn:x + w + cutn]  # 先用y确定高，再用x确定宽
    cv2.imshow("result", img)

def yucea(img,inname,slx,sly,srx,sry,delx,dely,derx,dery,imgray):  # 公式：savelo[x] = savelo[x-1] + s + de * ((inname - Z)//step)
    #根据加速度进行预测框选
    savelo[inname - start - 1] = [0, 0, 0, 0]
    yulx = savelo[inname - start - 2][0] + slx + delx *((inname - Z - 1)//step)
    yuly = savelo[inname - start - 2][1] + sly + dely *((inname - Z - 1)//step)
    yurx = savelo[inname - start - 2][2] + srx + derx *((inname - Z - 1)//step)
    yury = savelo[inname - start - 2][3] + sry + dery *((inname - Z - 1)//step)
    savelo[inname - start - 1] = [yulx, yuly, yurx, yury]
    cv2.rectangle(img, (yulx, yuly), (yurx, yury), (0, 255, 255), 2)  # 框选
    #根据边缘模板框选进行对比
    changex = 6
    changey = 6  # 放大边框x,y
    yuzhi = 100
    ret, thresh = cv2.threshold(imgray, yuzhi, 255, 0)  # 产生2值图
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image = img.copy()
    # print(contours)
    # cv2.imshow("erzhitu", thresh)
    #img = cv2.drawContours(img, contours, -1, (0, 255, 0), 5)  # img为三通道才能显示轮廓
    # cv2.imshow("drawCon",img)  #drawCon后的图像
    xmin = ymin = 10 ** 8
    xmax = ymax = 0
    for i in range(0, len(contours)):  # 确定框选坐标
        x, y, w, h = cv2.boundingRect(contours[i])
        if x < xmin: xmin = x
        if y < ymin: ymin = y
        if x + w > xmax: xmax = x + w
        if y + h > ymax: ymax = y + h
    if xmin > 0:  # 防止边界溢出等情况
        cv2.rectangle(img, (xmin, ymin), (xmax + changex, ymax + changey), (255, 0, 0), 2)  # 框选
        if inname < Z:
            push(xmin, ymin, xmax + changex, ymax + changey, inname)  # 把这次的坐标导入push中给savelo
        # newimage = image[y - cutn:y + h + cutn, x - cutn:x + w + cutn]  # 先用y确定高，再用x确定宽
    cv2.imshow("result", img)

def pinjun():
    #利用savelo预测，这里选择平均模型,savelo标注了之前模板匹配时用匹配图的左下和右上点的坐标信息
    print("取均值移动边框")
    avlx = avly = avrx = avry = 0
    for i in range(1,key - 1):
        avlx = avlx + savelo[i][0] - savelo[i-1][0]  #左下点x平均偏移
        avly = avly + savelo[i][1] - savelo[i-1][1]  #左下点y平均偏移
        avrx = avrx + savelo[i][2] - savelo[i-1][2]  #右上点x平均偏移
        avry = avry + savelo[i][3] - savelo[i-1][3]  #右上点y平均偏移
    avlx = round(avlx / (key-2))
    avly = round(avly / (key-2))
    avrx = round(avrx / (key-2))
    avry = round(avry / (key-2))
    return avlx,avly,avrx,avry

def jiasu():
    delx = dely = derx = dery = 0  # 拆分知道step时间加几,即成为加速度
    for i in range(1,key - 1):
        delx = (savelo[i][0] - savelo[i - 1][0] - slx) + delx
        dely = (savelo[i][1] - savelo[i - 1][1] - sly) + dely
        derx = (savelo[i][2] - savelo[i - 1][2] - srx) + derx
        dery = (savelo[i][3] - savelo[i - 1][3] - sry) + dery
    print("sl")
    print(slx,sly,srx,sry)
    print("desum")
    print(delx,dely,derx,dery)
    delx = abs(round(delx / step))
    dely = abs(round(dely / step))
    derx = abs(round(derx / step))
    dery = abs(round(dery / step))
    print("deav")
    print(delx,dely,derx,dery)
    return delx,dely,derx,dery

if __name__ == "__main__":
    img = cv2.imread(addpath(inname))  # 图片
    template = cv2.imread("./shiyan4/temp.bmp")
    dstback = np.zeros((img.shape[0],img.shape[1]))
    key = Z
    Z = Z + inname  #分割序列号
    start = p = inname   #初始输入序列号start不会改变，其余会变
    savelo = {}
    while inname<Z:
        print(inname - start)
        img = cv2.imread(addpath(inname))  # 图片
        template = cv2.imread(muname)  # 模板
        grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        graytemplate = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        #cannymatch(grayimg)
        dst = dajinmatch(grayimg) #大津算法出边缘
        imgwhite = xiangjian(dst,dstback) #出二值图
        kuangxuan(imgwhite,img,inname,Z)
        dstback = dst  #导致第一帧没有录入框
        #matchSQDIFF(grayimg, graytemplate) #标准平方差匹配
        #matchCCORR(grayimg,graytemplate) #标准相关匹配
        #matchSIFT(grayimg, graytemplate)  #sift匹配
        inname = inname + 1
    avlx,avly,avrx,avry = pinjun()    #利用savelo预测，这里选择平均模型,savelo标注了之前模板匹配时用匹配图的左下和右上点的坐标信息
    slx,sly,srx,sry = savelo[key - 2]-savelo[key - 3]  #设置对于之后帧初始的每格位移 start left x 等等
    delx = dely = derx = dery = 0  # 拆分知道step时间加几,即成为加速度
    delx,dely,derx,dery = jiasu()
    #公式：savelo[x] = savelo[x-1] + s + de * [(inname - Z)//step] 下面求de
    while (inname>=Z) and (inname < 497):
        img = cv2.imread(addpath(inname))  # 图片
        template = cv2.imread(muname)  # 模板
        grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        graytemplate = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        # cannymatch(grayimg)
        dst = dajinmatch(grayimg)
        imgwhite = xiangjian(dst, dstback)
        #yucemean(img,inname)   #平均值预测
        yucea(img,inname,slx,sly,srx,sry,delx,dely,derx,dery,imgwhite)   #加速度预测
        #kuangxuan(imgwhite,img,inname,Z)
        dstback = dst
        # matchSQDIFF(grayimg, graytemplate) #标准平方差匹配
        # matchCCORR(grayimg,graytemplate) #标准相关匹配
        # matchSIFT(grayimg, graytemplate)  #sift匹配
        inname = inname + 1
    cv2.waitKey(0)