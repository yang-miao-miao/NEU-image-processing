import numpy as np
import cv2 as cv
def rgb2gray(img):
    h=img.shape[0]
    w=img.shape[1]
    img1=np.zeros((h,w),np.uint8)
    for i in range(h):
        for j in range(w):
            #img1[i,j]=0.144*img[i,j,0]+0.587*img[i,j,1]+0.299*img[i,j,2]
            img1[i, j] = img[i, j, 0]
    return img1
#1x5
def median1(img):
    h=img.shape[0]
    w=img.shape[1]
    img1 = np.zeros((h, w), np.uint8)
    for i in range (2,h-2):
        for j in range (0,w):
            temporary = np.zeros(5, np.uint8)
            s=0
            for k in range (-2,3):
                for l in range (0,1):
                    temporary[s]=img[i+k,j+l]
                    s+=1
            temporary=np.sort(temporary)
            median=temporary[2]
            img1[i,j]=median
    return img1
#5x1
def median2(img):
    h=img.shape[0]
    w=img.shape[1]
    img1 = np.zeros((h, w), np.uint8)
    for i in range (0,h):
        for j in range (2,w-2):
            temporary = np.zeros(5, np.uint8)
            s=0
            for k in range (0,1):
                for l in range (-2,3):
                    temporary[s]=img[i+k,j+l]
                    s+=1
            temporary=np.sort(temporary)
            median=temporary[2]
            img1[i,j]=median
    return img1
#3x3十字
def median3(img):
    h=img.shape[0]
    w=img.shape[1]
    img1 = np.zeros((h, w), np.uint8)
    for i in range (1,h-1):
        for j in range (1,w-1):
            temporary = np.zeros(5, np.uint8)
            s=0
            for k in range (-1,0):
                for l in range (0,1):
                    temporary[s]=img[i+k,j+l]
                    s+=1
            for k in range (0,1):
                for l in range (-1,2):
                    temporary[s]=img[i+k,j+l]
                    s+=1
            for k in range (1,2):
                for l in range (0,1):
                    temporary[s]=img[i+k,j+l]
                    s+=1
            temporary=np.sort(temporary)
            median=temporary[2]
            img1[i,j]=median
    return img1
#5x5十字
def median4(img):
    h=img.shape[0]
    w=img.shape[1]
    img1 = np.zeros((h, w), np.uint8)
    for i in range (2,h-2):
        for j in range (2,w-2):
            temporary = np.zeros(9, np.uint8)
            s=0
            for k in range (-2,0):
                for l in range (0,1):
                    temporary[s]=img[i+k,j+l]
                    s+=1
            for k in range (0,1):
                for l in range (-2,3):
                    temporary[s]=img[i+k,j+l]
                    s+=1
            for k in range (1,3):
                for l in range (0,1):
                    temporary[s]=img[i+k,j+l]
                    s+=1
            temporary=np.sort(temporary)
            median=temporary[4]
            img1[i,j]=median
    return img1
#3x3方形
def median5(img):
    h=img.shape[0]
    w=img.shape[1]
    img1 = np.zeros((h, w), np.uint8)
    for i in range (1,h-1):
        for j in range (1,w-1):
            temporary = np.zeros(9, np.uint8)
            s=0
            for k in range (-1,2):
                for l in range (-1,2):
                    temporary[s]=img[i+k,j+l]
                    s+=1
            temporary=np.sort(temporary)
            median=temporary[4]
            img1[i,j]=median
    return img1
#5x5菱形
def median6(img):
    h=img.shape[0]
    w=img.shape[1]
    img1 = np.zeros((h, w), np.uint8)
    for i in range (2,h-2):
        for j in range (2,w-2):
            temporary = np.zeros(13, np.uint8)
            s=0
            for k in range (-2,-1):
                for l in range (0,1):
                    temporary[s]=img[i+k,j+l]
                    s+=1
            for k in range (-1,0):
                for l in range (-1,2):
                    temporary[s]=img[i+k,j+l]
                    s+=1
            for k in range (0,1):
                for l in range (-2,3):
                    temporary[s]=img[i+k,j+l]
                    s+=1
            for k in range (1,2):
                for l in range (-1,2):
                    temporary[s]=img[i+k,j+l]
                    s+=1
            for k in range (2,3):
                for l in range (0,1):
                    temporary[s]=img[i+k,j+l]
                    s+=1
            temporary=np.sort(temporary)
            median=temporary[6]
            img1[i,j]=median
    return img1
#7x7菱形
def median7(img):
    h=img.shape[0]
    w=img.shape[1]
    img1 = np.zeros((h, w), np.uint8)
    for i in range (3,h-3):
        for j in range (3,w-3):
            temporary = np.zeros(25, np.uint8)
            s=0
            for k in range (-3,-2):
                for l in range (0,1):
                    temporary[s]=img[i+k,j+l]
                    s+=1
            for k in range (-2,-1):
                for l in range (-1,2):
                    temporary[s]=img[i+k,j+l]
                    s+=1
            for k in range (-1,0):
                for l in range (-2,3):
                    temporary[s]=img[i+k,j+l]
                    s+=1
            for k in range (0,1):
                for l in range (-3,4):
                    temporary[s]=img[i+k,j+l]
                    s+=1
            for k in range (1,2):
                for l in range (-2,3):
                    temporary[s]=img[i+k,j+l]
                    s+=1
            for k in range (2,3):
                for l in range (-1,2):
                    temporary[s]=img[i+k,j+l]
                    s+=1
            for k in range (3,4):
                for l in range (0,1):
                    temporary[s]=img[i+k,j+l]
                    s+=1
            temporary=np.sort(temporary)
            median=temporary[12]
            img1[i,j]=median
    return img1
#5x5圆形
def median8(img):
    h=img.shape[0]
    w=img.shape[1]
    img1 = np.zeros((h, w), np.uint8)
    for i in range (2,h-2):
        for j in range (2,w-2):
            temporary = np.zeros(21, np.uint8)
            s=0
            for k in range (-2,-1):
                for l in range (-1,2):
                    temporary[s]=img[i+k,j+l]
                    s+=1
            for k in range (-1,2):
                for l in range (-2,3):
                    temporary[s]=img[i+k,j+l]
                    s+=1
            for k in range (2,3):
                for l in range (-1,2):
                    temporary[s]=img[i+k,j+l]
                    s+=1
            temporary=np.sort(temporary)
            median=temporary[10]
            img1[i,j]=median
    return img1
#7x7圆形
def median9(img):
    h=img.shape[0]
    w=img.shape[1]
    img1 = np.zeros((h, w), np.uint8)
    for i in range (3,h-3):
        for j in range (3,w-3):
            temporary = np.zeros(37, np.uint8)
            s=0
            for k in range (-3,-2):
                for l in range (-1,2):
                    temporary[s]=img[i+k,j+l]
                    s+=1
            for k in range (-2,-1):
                for l in range (-2,3):
                    temporary[s]=img[i+k,j+l]
                    s+=1
            for k in range (-1,2):
                for l in range (-3,4):
                    temporary[s]=img[i+k,j+l]
                    s+=1
            for k in range (2,3):
                for l in range (-2,3):
                    temporary[s]=img[i+k,j+l]
                    s+=1
            for k in range (3,4):
                for l in range (-1,2):
                    temporary[s]=img[i+k,j+l]
                    s+=1
            temporary=np.sort(temporary)
            median=temporary[18]
            img1[i,j]=median
    return img1
img=cv.imread('./images/1/boy_noisy.jpg')
gray=rgb2gray(img)
img1=median1(gray)
img2=median2(gray)
img3=median3(gray)
img4=median4(gray)
img5=median5(gray)
img6=median6(gray)
img7=median7(gray)
img8=median8(gray)
img9=median9(gray)
'''cv.imwrite('./imgwork/learn/que1/Wirebond/img1.tif',img1)
cv.imwrite('./imgwork/learn/que1/Wirebondn/img2.tif',img2)
cv.imwrite('./imgwork/learn/que1/Wirebond/img3.tif',img3)
cv.imwrite('./imgwork/learn/que1/Wirebond/img4.tif',img4)
cv.imwrite('./imgwork/learn/que1/Wirebond/img5.tif',img5)
cv.imwrite('./imgwork/learn/que1/Wirebond/img6.tif',img6)
cv.imwrite('./imgwork/learn/que1/Wirebond/img7.tif',img7)
cv.imwrite('./imgwork/learn/que1/Wirebond/img8.tif',img8)
cv.imwrite('./imgwork/learn/que1/Wirebond/img9.tif',img9)'''
cv.imshow('img',img)
cv.imshow('img1',img1)
cv.imshow('img2',img2)
cv.imshow('img3',img3)
cv.imshow('img4',img4)
cv.imshow('img5',img5)
cv.imshow('img6',img6)
cv.imshow('img7',img7)
cv.imshow('img8',img8)
cv.imshow('img9',img9)
cv.waitKey(0)
cv.destroyAllWindows()