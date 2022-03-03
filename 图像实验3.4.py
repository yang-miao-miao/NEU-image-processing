import cv2 as cv
import numpy as np
import cv2
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号
img=cv.imread('./shiyan3/object1.png',0)
plt.subplot(2,2,1)
plt.title('原图')
plt.imshow(img,cmap='gray')
rows,cols=img.shape
#旋转中心，旋转角度，缩放比例
M=cv.getRotationMatrix2D((cols / 2, rows / 2), 45, 0.6)
roll=cv.warpAffine(img, M, (cols, rows))
plt.subplot(2,2,2)
plt.title('旋转')
plt.imshow(roll,cmap='gray')
#放大
big=cv.resize(img,None,fx=2,fy=2)
plt.subplot(2,2,3)
plt.title('放大')
plt.imshow(big,cmap='gray')
#镜像
jingxiang = cv2.flip(img,1)  #镜像
plt.subplot(2,2,4)
plt.title('镜像')
plt.imshow(jingxiang,cmap='gray')
plt.show()
#不变矩特征
def mon(img):
    y = []
    for i in range(cols):
        for j in range(1, rows + 1):
            y.append(j)
    x = []
    for i in range(1, cols + 1):
        for j in range(1, rows + 1):
            x.append(i)
    for i in range(rows):
        for j in range(cols):
            if img[i, j] > 200:
                img[i, j] = 1
    A = []
    for i in range(rows):
        for j in range(cols):
            A.append(img[i, j])
    m00 = sum(A)
    m10 = sum([i * j for i, j in zip(x, A)])
    m01 = sum([i * j for i, j in zip(y, A)])
    xmean = m10 / m00
    ymean = m01 / m00
    cm00 = m00
    cm02 = sum([i * j for i, j in zip((y - ymean) ** 2, A)] / (m00 ** 2))
    cm03 = sum([i * j for i, j in zip((y - ymean) ** 3, A)] / (m00 ** 2.5))
    cm20 = sum([i * j for i, j in zip((x - xmean) ** 2, A)] / (m00 ** 2))
    cm30 = sum([i * j for i, j in zip((x - xmean) ** 3, A)] / (m00 ** 2.5))
    cm11 = sum([i * j * k for i, j, k in zip((x - xmean), (y - ymean), A)]) / (m00 ** 2)
    cm12 = sum([i * j * k for i, j, k in zip((x - xmean), (y - ymean) ** 2, A)]) / (m00 ** 2.5)
    cm21 = sum([i * j * k for i, j, k in zip((x - xmean) ** 2, (y - ymean), A)]) / (m00 ** 2.5)
    Mon1 = cm20 + cm02  # 1阶矩Mon(1)
    Mon2 = (cm20 - cm02) ** 2 + 4 * cm11 ** 2  # 2阶矩Mon(2)
    Mon3 = (cm30 - 3 * cm12) ** 2 + (3 * cm21 - cm03) ** 2  # 3阶矩Mon(3)
    Mon4 = (cm30 + cm12) ** 2 + (cm21 + cm03) ** 2  # 4阶矩Mon(4)
    Mon5 = (cm30 - 3 * cm12) * (cm30 + cm12) * ((cm30 + cm12) ** 2 - 3 * (cm21 + cm03) ** 2) + (
                3 * (cm30 + cm12) ** 2 - (cm21 + cm03) ** 2);  # 5阶矩Mon(5)
    Mon6 = (cm20 - cm02) * ((cm30 + cm12) ** 2 - (cm21 + cm03) ** 2) + 4 * cm11 * (cm30 + cm12) * (
                cm21 + cm03);  # 6阶矩Mon(6)
    Mon7 = (3 * cm21 - cm03) * (cm30 + cm12) * ((cm30 + cm12) ** 2 - 3 * (cm21 + cm03) ** 2) + (3 * cm12 - cm30) * (
                cm21 + cm03) * (3 * (cm30 + cm12) ** 2 - (cm21 + cm03) ** 2);  # 7阶矩Mon(7)
    Mon = [Mon1, Mon2, Mon3, Mon4, Mon5, Mon6, Mon7]
    print(Mon)
print('img')
mon(img)
print('roll')
mon(roll)
print('big')
mon(big)
print('jingxiang')
mon(jingxiang)
cv.waitKey(0)
cv.destroyAllWindows()