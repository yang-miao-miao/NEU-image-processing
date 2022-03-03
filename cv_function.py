import cv2 as cv
import cv2
import random
import matplotlib.pyplot as plt
import numpy as np
'''实验一'''
# 1.1图像对比度增强变换  y=x*k
def linear_transform(input,k,output):
    height = input.shape[0]
    width = input.shape[1]
    for i in range(height):
        for j in range(width):
            if (int(input[i, j] * k) > 255):
                gray = 255
            else:
                gray = int(input[i, j] * k)
            output[i, j] = np.uint8(gray)
    return output
# 1.2对数变换
def log_transform(c, img):
    output = c * np.log(1.0 + img)
    output = np.uint8(output + 0.5)
    return output
# 1.3直方图均衡
def equalize(img):
    equ = cv.equalizeHist(img)
    hist1 = cv.calcHist([img], [0], None, [256], [0, 256])
    hist2 = cv.calcHist([equ], [0], None, [256], [0, 256])
    plt.plot(hist1)
    plt.plot(hist2)
    #plt.show()
    return equ
# 2.1添加椒盐噪声    prob:噪声比例
def jiaoyan_noise(image,prob):
    output = np.zeros(image.shape,np.uint8)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output
# 2.2添加高斯噪声
def gaosi_noise(image, mean, var):
    image = np.array(image/255, dtype=float)#归一化
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    out = image + noise
    if out.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    out = np.clip(out, low_clip, 1.0)
    out = np.uint8(out*255)
    return out
# 3.1 理想低通频域滤波
def lixiangditong(img,r): #r为半径
    #opencv中的傅立叶变换
    dft=cv.dft(np.float32(img), flags=cv.DFT_COMPLEX_OUTPUT)
    dtf_shift=np.fft.fftshift(dft) #np.fft.fftshift()函数来实现平移,让直流分量在输出图像的重心
    rows,cols=img.shape
    crow,ccol=int(rows/2),int(cols/2) #计算频谱中心
    mask=np.zeros((rows,cols,2),np.uint8) #生成rows行cols列的2纬矩阵，数据格式为uint8
    mask[crow-r:crow+r,ccol-r:ccol+r]=1 #将靠近频谱中心的部分低通信息 设置为1，属于低通滤波
    fshift=dtf_shift*mask
    #傅立叶逆变换
    f_ishift=np.fft.ifftshift(fshift)
    img_back=cv.idft(f_ishift)
    img_back=cv.magnitude(img_back[:, :, 0], img_back[:, :, 1]) #计算像素梯度的绝对值
    img_back=np.abs(img_back)
    img_back=(img_back-np.amin(img_back))/(np.amax(img_back)-np.amin(img_back))
    return img_back
# 3.2 巴特沃斯滤波器
def batewosi(image, d, n):
    f = np.fft.fft2(image)
    # 取绝对值后将复数变化为实数
    # 取对数的目的是将数据变换到0~255
    fshift = np.fft.fftshift(f)
    s1 = np.log(np.abs(fshift))
    def make_transform_matrix(d):
        transform_matrix = np.zeros(image.shape)
        center_point = tuple(map(lambda x: (x - 1) / 2, s1.shape))
        for i in range(transform_matrix.shape[0]):
            for j in range(transform_matrix.shape[1]):
                def cal_distance(pa, pb):
                    from math import sqrt
                    dis = sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
                    return dis
                dis = cal_distance(center_point, (i, j))
                transform_matrix[i, j] = 1 / (1 + (dis / d) ** (2 * n))
        return transform_matrix
    d_matrix = make_transform_matrix(d)
    new_img = np.abs(np.fft.ifft2(np.fft.ifftshift(fshift * d_matrix)))
    return new_img
# 3.3 高斯低通频域滤波
def gaosi(img, sigma=90):
    imarr = np.array(img)
    height, width = imarr.shape
    fft = np.fft.fft2(imarr)
    fft = np.fft.fftshift(fft)
    for i in range(height):
        for j in range(height):
            fft[i, j] *= np.exp(-((i - (height - 1) / 2) ** 2 + (j - (width - 1) / 2) ** 2) / 2 / sigma ** 2)
    fft = np.fft.ifftshift(fft)
    ifft = np.fft.ifft2(fft)
    ifft = np.real(ifft)
    max = np.max(ifft)
    min = np.min(ifft)
    res = np.zeros((height, width), dtype="uint8")
    for i in range(height):
        for j in range(width):
            res[i, j] = 255 * (ifft[i, j] - min) / (max - min)
    return res
# 4.去雾增强相关函数
def zmMinFilterGray(src, r=7):
    '''最小值滤波，r是滤波器半径'''
    return cv.erode(src, np.ones((2 * r + 1, 2 * r + 1)))
def guidedfilter(I, p, r, eps):
    height, width = I.shape
    m_I = cv.boxFilter(I, -1, (r, r))
    m_p = cv.boxFilter(p, -1, (r, r))
    m_Ip = cv.boxFilter(I * p, -1, (r, r))
    cov_Ip = m_Ip - m_I * m_p
    m_II = cv.boxFilter(I * I, -1, (r, r))
    var_I = m_II - m_I * m_I
    a = cov_Ip / (var_I + eps)
    b = m_p - a * m_I
    m_a = cv.boxFilter(a, -1, (r, r))
    m_b = cv.boxFilter(b, -1, (r, r))
    return m_a * I + m_b
def Defog(m, r, eps, w, maxV1):                 # 输入rgb图像，值范围[0,1]
    '''计算大气遮罩图像V1和光照值A, V1 = 1-t/A'''
    V1 = np.min(m, 2)                           # 得到暗通道图像
    Dark_Channel = zmMinFilterGray(V1, 7)
    V1 = guidedfilter(V1, Dark_Channel, r, eps)  # 使用引导滤波优化
    bins = 2000
    ht = np.histogram(V1, bins)                  # 计算大气光照A
    d = np.cumsum(ht[0]) / float(V1.size)
    for lmax in range(bins - 1, 0, -1):
        if d[lmax] <= 0.999:
            break
    A = np.mean(m, 2)[V1 >= ht[1][lmax]].max()
    V1 = np.minimum(V1 * w, maxV1)               # 对值范围进行限制
    return V1, A
def deHaze(m, r=81, eps=0.001, w=0.98, maxV1=0.80, bGamma=False):
    Y = np.zeros(m.shape)
    Mask_img, A = Defog(m, r, eps, w, maxV1)             # 得到遮罩图像和大气光照
    for k in range(3):
        Y[:,:,k] = (m[:,:,k] - Mask_img)/(1-Mask_img/A)  # 颜色校正
    Y = np.clip(Y, 0, 1)
    if bGamma:
        Y = Y ** (np.log(0.5) / np.log(Y.mean()))       # gamma校正,默认不进行该操作
    return Y


'''实验二'''
#2.1迭代阈值分割
def diedai(img):
    img_array = np.array(img).astype(np.float32)#转化成数组
    I=img_array
    zmax=np.max(I)
    zmin=np.min(I)
    tk=(zmax+zmin)/2#设置初始阈值
    #根据阈值将图像进行分割为前景和背景，分别求出两者的平均灰度  zo和zb
    b=1
    m,n=I.shape
    while b==0:
        ifg=0
        ibg=0
        fnum=0
        bnum=0
        for i in range(1,m):
             for j in range(1,n):
                tmp=I[i,j]
                if tmp>=tk:
                    ifg=ifg+1
                    fnum=fnum+int(tmp)  #前景像素的个数以及像素值的总和
                else:
                    ibg=ibg+1
                    bnum=bnum+int(tmp)#背景像素的个数以及像素值的总和
        #计算前景和背景的平均值
        zo=int(fnum/ifg)
        zb=int(bnum/ibg)
        if tk==int((zo+zb)/2):
            b=0
        else:
            tk=int((zo+zb)/2)
    print('阈值为:',tk)
    ret2, th2 = cv2.threshold(img, tk, 255, cv2.THRESH_BINARY)
    return th2

#2.2八方向的边缘检测
def eight_canny(img,i):
    def conv_cal(img, filter):
        h, w = img.shape
        img_filter = np.zeros([h, w])
        for i in range(h - 2):
            for j in range(w - 2):
                img_filter[i][j] = img[i][j] * filter[0][0] + img[i][j + 1] * filter[0][1] + img[i][j + 2] * filter[0][
                    2] + \
                                   img[i + 1][j] * filter[1][0] + img[i + 1][j + 1] * filter[1][1] + img[i + 1][j + 2] * \
                                   filter[1][2] + \
                                   img[i + 2][j] * filter[2][0] + img[i + 2][j + 1] * filter[2][1] + img[i + 2][j + 2] * \
                                   filter[2][2]
        return img_filter
    krisch1 = np.array([[5, 5, 5],
                        [-3, 0, -3],
                        [-3, -3, -3]])
    krisch2 = np.array([[-3, -3, -3],
                        [-3, 0, -3],
                        [5, 5, 5]])
    krisch3 = np.array([[5, -3, -3],
                        [5, 0, -3],
                        [5, -3, -3]])
    krisch4 = np.array([[-3, -3, 5],
                        [-3, 0, 5],
                        [-3, -3, 5]])
    krisch5 = np.array([[-3, -3, -3],
                        [-3, 0, 5],
                        [-3, 5, 5]])
    krisch6 = np.array([[-3, -3, -3],
                        [5, 0, -3],
                        [5, 5, -3]])
    krisch7 = np.array([[-3, 5, 5],
                        [-3, 0, 5],
                        [-3, -3, -3]])
    krisch8 = np.array([[5, 5, -3],
                        [5, 0, -3],
                        [-3, -3, -3]])
    kernel=[krisch1,krisch2,krisch3,krisch4,krisch5,krisch6,krisch7,krisch8]
    w, h = img.shape
    img2 = np.zeros([w + 2, h + 2])
    img2[2:w + 2, 2:h + 2] = img[0:w, 0:h]
    edge = conv_cal(img2, kernel[i])
    # for i in range(w):
    #   for j in range(h):
    #      edge_img[i][j]=max(list([edge1[i][j],edge2[i][j],edge3[i][j],edge4[i][j],edge5[i][j],edge6[i][j],edge7[i][j],edge8[i][j]]))
    return edge

#2.3霍夫圆检测分割
def circle_detect(img):
    img2=img.copy()
    '''minDist表示两个圆之间圆心的最小距离(重要)
    param1有默认值100，它是method设置的检测方法的对应的参数，对当前唯一的方法霍夫梯度法cv2.HOUGH_GRADIENT，它表示传递给canny边缘检测算子的高阈值，而低阈值为高阈值的一半
    param2有默认值100，它是method设置的检测方法的对应的参数，对当前唯一的方法霍夫梯度法cv2.HOUGH_GRADIENT，它表示在检测阶段圆心的累加器阈值，它越小，就越可以检测到更多根本不存在的圆，而它越大的话，能通过检测的圆就更加接近完美的圆形了
    minRadius有默认值0，圆半径的最小
    maxRadius有默认值0，圆半径的最大值'''
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 50, param1=150, param2=60, minRadius=5, maxRadius=70)
    circles = np.uint16(np.around(circles))
    circles_count = 0
    for i in circles[0, :]:
        cv2.circle(img2, (i[0], i[1]), i[2], (255, 0, 0), 2)
        circles_count = 1 + circles_count
        cv2.circle(img2, (i[0], i[1]), 2, (0, 0, 255), 1)
        for m in range(img2.shape[0]):
            for n in range(img2.shape[1]):
                if (m-i[1])**2+(n-i[0])**2<i[2]**2:
                    img2[m,n]=0
        for m in range(img2.shape[0]):
            for n in range(img2.shape[1]):
                if img2[m,n]!=0:
                    img2[m,n]=255
    print("圆形个数:", circles_count)
    return img2
#dowels检测
dowels=cv2.imread('E:/Desktop/shiyan2/Dowels.tif',0)
def dowels_detect(img):
    gray = img.copy()
    # 使用局部阈值的自适应阈值操作进行图像二值化
    ret, thresh = cv.threshold(gray, 100, 255, cv.THRESH_BINARY)
    kernel = np.ones((10, 10), np.uint8)
    dst = cv.erode(thresh, kernel)
    contours, hierarchy = cv2.findContours(dst, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(dst, contours, -1, (120, 0, 0), 2)
    count = 0  # 米粒总数
    ares_avrg = 0  # 米粒平均
    for cont in contours:
        ares = cv2.contourArea(cont)
        if ares < 50:
            continue
        count += 1
        ares_avrg += ares
        print("{}-blob:{}".format(count, ares), end="  ")
        rect = cv2.boundingRect(cont)
        print("x:{} y:{}".format(rect[0], rect[1]))
        cv2.rectangle(img, rect, (0, 0, 255), 1)
        # 防止编号到图片之外（上面）,因为绘制编号写在左上角，所以让最上面的米粒的y小于10的变为10个像素
        y = 10 if rect[1] < 10 else rect[1]
        # 在米粒左上角写上编号
        cv2.putText(img, str(count), (rect[0], y), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0, 255, 0), 1)
    print('个数', count, ' 总面积', ares_avrg, ' ares', ares)
    print("米粒平均面积:{}".format(round(ares_avrg / count, 2)))  # 打印出每个米粒的面积v
#米粒检测
def rice_detect(img):
    gray = img.copy()
    # 使用局部阈值的自适应阈值操作进行图像二值化
    dst = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 101, 1)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    dst = cv2.morphologyEx(dst, cv2.MORPH_OPEN, element)
    contours, hierarchy = cv2.findContours(dst, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(dst, contours, -1, (120, 0, 0), 2)
    count = 0  # 米粒总数
    ares_avrg = 0  # 米粒平均
    for cont in contours:
        ares = cv2.contourArea(cont)
        if ares < 50:
            continue
        count += 1
        ares_avrg += ares
        print("{}-blob:{}".format(count, ares), end="  ")
        rect = cv2.boundingRect(cont)
        print("x:{} y:{}".format(rect[0], rect[1]))
        cv2.rectangle(img, rect, (0, 0, 255), 1)
        # 防止编号到图片之外（上面）,因为绘制编号写在左上角，所以让最上面的米粒的y小于10的变为10个像素
        y = 10 if rect[1] < 10 else rect[1]
        # 在米粒左上角写上编号
        cv2.putText(img, str(count), (rect[0], y), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0, 255, 0), 1)
    print('个数', count, ' 总面积', ares_avrg, ' ares', ares)
    print("米粒平均面积:{}".format(round(ares_avrg / count, 2)))  # 打印出每个米粒的面积