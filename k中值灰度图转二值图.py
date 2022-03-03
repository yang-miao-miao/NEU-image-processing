import matplotlib.pyplot as plt
from random import  sample
import pandas as pd
import numpy as np
from PIL import Image          #导入必要的python包
def get_pixel(grey_image):
    #获取图像像素
    pixel = []
    for i in range(grey_image.size[1]):
        for j in range(grey_image.size[0]):
            x = grey_image.getpixel((j,i))  # 获取图片的每一个像素  (i,j)(i,j)  的 RBG 值
            pixel.append(x)
    return pixel
def k_median(pixel,k):
    #k-means算法具体实现
    C=sample(pixel,2)
    error=10e3
    while error>10e-10:
        D = np.zeros((len(pixel), k))  #D为样本到每一个中心的距离平方，
        for i in range(k):             #对两个中心点开始循环
            cc=[]                      #建立空列表
            cc.append(C[i])            #将每次提取的两个数据中挨个放入cc
            cc=np.array(cc)            #形成numpy数组
            pixel=np.array(pixel)      #形成numpy数组
            cc=np.full((len(pixel)), cc) #填充pixel长度个cc用于后期计算
            D[:, i] = np.square(pixel - cc)#计算聚类中心到每个点的欧式距离
        labels = np.argmin(D, axis=1)    #给出最小值坐标
        pix_dataFrame=pd.DataFrame(pixel)  #形成DataFrame
        C_2=pix_dataFrame.groupby(labels).median() #用中值计算新的聚类中心
        C=pd.DataFrame(C)                  #形成DataFrame
        error = np.linalg.norm(C_2 - C)    #计算误差
        C=np.array(C)                      #形成numpy数组
        C_2=np.array(C_2)                  #形成numpy数组
        C=C_2                              #将新的计算中心赋值给C，这样构成循环，直到在误差范围内后输出
    return labels,C
#读取数据
path ='img1.jpg'
I = Image.open(path)
m,n=I.size
L = I.convert('L') #转化为灰度值
#获取像素值
pixel=get_pixel(L)
print(np.shape(pixel))
k=2    #k为聚类数目
#k-means算法实现
labels,C =k_median(pixel,k)
#分割图labels图像显示
labels = labels.reshape(n,m)
plt.imshow(labels,cmap='gray')
#plt.savefig('img1.1.jpg')
plt.show()