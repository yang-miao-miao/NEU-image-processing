import cv2
import numpy as np
import math
from matplotlib import pyplot as plt

name = './shiyan3/object1.png'
name1 = './shiyan3/object2.png'

def Euler(img): #计算欧拉数
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 80, 255, 0)  # 产生2值图
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #print(hierarchy) #后一个轮廓、前一个轮廓、父轮廓、内嵌轮廓的索引编号,如果没有对应项，则该值为负数。
    m = 0 #欧拉计数器
    #print(len(hierarchy[0]))
    for i in range(len(hierarchy[0])):
        if hierarchy[0][i][3] == 0:
            m=m+1
    return 1-m  #连通区域为1

def simS(img):  #计算面积
    """height,width = img.shape[:2]
    S = np.sum(img == 255)   #因为是二值图，又取白色为需要颜色，所以白色像素为面积"""
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 80, 255, 0)  # 产生2值图
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    S = cv2.contourArea(contours[0])  #自带面积函数
    return S

def simR0(img): #计算圆形度
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 80, 255, 0)  # 产生2值图
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    a = cv2.contourArea(contours[0]) * 4 * math.pi
    b = math.pow(cv2.arcLength(contours[0], True), 2)
    if b == 0:
        return 0
    return a / b

def sime(img):  #计算形状复杂度
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 80, 255, 0)  # 产生2值图
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    a = cv2.contourArea(contours[0])
    b = math.pow(cv2.arcLength(contours[0], True), 2)
    if b == 0:
        return 0
    return b / a

def simC(img):  #计算周长
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 80, 255, 0)  # 产生2值图
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    imgthick = thresh.copy()
    cv2.drawContours(imgthick, contours,-1,(100,100,100),5,lineType=cv2.LINE_AA)
    cv2.imshow('Contours', imgthick) #画出周长对应的线
    C = cv2.arcLength(contours[0], True)  #自带周长函数
    #print("C",cv2.contourArea(contours[0]))
    """C = 0   #当作多端线段计算
    #print(contours[0][0][0][0])  #这是第一位的第一个数字
    for i in range(len(contours[0])-1):
        C = np.sqrt((contours[0][i][0][0]-contours[0][i+1][0][0]) ** 2 + (contours[0][i][0][1]-contours[0][i+1][0][1])   ** 2) + C
        #print (contours[0][i][0])
    C = np.sqrt((contours[0][0][0][0] - contours[0][len(contours)-1][0][0]) ** 2 + (contours[0][0][0][1] - contours[0][len(contours)-1][0][1]) ** 2) + C"""
    return C

if __name__ == "__main__":
    img = cv2.imread(name)
    img1 = cv2.imread(name1)
    #a
    print("{}欧拉数为：".format(name),Euler(img))
    print("{}面积为：".format(name),simS(img))
    print("{}周长为：".format(name),simC(img))
    print("{}圆形度为：".format(name),simR0(img))
    print("{}形状复杂度为：".format(name),sime(img))
    #b
    print("{}欧拉数为：".format(name1),Euler(img1))
    print("{}面积为：".format(name1),simS(img1))
    print("{}周长为：".format(name1),simC(img1))
    print("{}圆形度为：".format(name1),simR0(img1))
    print("{}形状复杂度为：".format(name1),sime(img1))

    cv2.waitKey(0)
    cv2.destroyAllWindows()