import cv2
import numpy as np
import matplotlib.pyplot as plt

muname = "./shiyan4/temp.bmp"
inname = 473

def addpath(name):  #方便图像序列增加
    cname = "./shiyan4/pic/" + str(name) + ".bmp"
    return cname

def matchSQDIFF(img,template):  #标准平方差，后面效果差
    th,tw = template.shape[:2]
    rv = cv2.matchTemplate(img,template,cv2.TM_SQDIFF_NORMED)  #标准平方差匹配
    minVal, maxVal,minLoc,maxLoc = cv2.minMaxLoc(rv)  #求矩阵的最小值，最大值和其索引
    topLeft = minLoc  #最小值是最关联部分，索引为位置坐标
    bottomRight = (topLeft[0]+tw,topLeft[1]+th)
    cv2.rectangle(img,topLeft,bottomRight,255,2)
    cv2.imshow("kuangchu",img)
    cv2.waitKey(70)

def matchCCORR(img,template):  #相关系数匹配,效果差
    th,tw = template.shape[:2]
    rv = cv2.matchTemplate(img,template,cv2.TM_SQDIFF_NORMED)  #标准相关匹配
    minVal, maxVal,minLoc,maxLoc = cv2.minMaxLoc(rv)  #求矩阵的最小值，最大值和其索引
    topLeft = maxLoc  #最大值是最关联部分，索引为位置坐标
    bottomRight = (topLeft[0]+tw,topLeft[1]+th)
    cv2.rectangle(img,topLeft,bottomRight,255,2)
    cv2.imshow("kuangchu",img)
    cv2.waitKey(70)

def matchSIFT(img,template):   #sift
    MIN_MATCH_COUNT = 6  # 设置最低特征点匹配数量为
    target = img
    #创建sift检测器
    sift =cv2.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(template, None)
    kp2, des2 = sift.detectAndCompute(target, None)
    # 创建设置FLANN匹配
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    # 存储所有好的匹配
    good = []
    # 舍弃大于0.8的匹配,只保留峰值大于主方向峰值80％的方向作为该关键点的辅方向。
    for m, n in matches:
        if m.distance < 0.8 * n.distance:
            good.append(m)
    if len(good) > MIN_MATCH_COUNT:
        # 获取关键点的坐标
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        # 计算变换矩阵和MASK
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()
        h, w = template.shape
        # 使用得到的变换矩阵对原图像的四个角进行变换，获得在目标图像上对应的坐标
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)
        cv2.polylines(target, [np.int32(dst)], True, 0, 2, cv2.LINE_AA)
    else:
        print("Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT))
        matchesMask = None
    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=None,
                       matchesMask=matchesMask,
                       flags=2)
    result = cv2.drawMatches(template, kp1, target, kp2, good, None, **draw_params)
    cv2.imshow("kuangchu",result)
    cv2.waitKey(70)


if __name__ == "__main__":
    while inname<497:
        img = cv2.imread(addpath(inname))  # 图片
        template = cv2.imread(muname)  # 模板
        grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        graytemplate = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        #matchSQDIFF(grayimg, graytemplate) #标准平方差匹配
        #matchCCORR(grayimg,graytemplate) #标准相关匹配
        matchSIFT(grayimg, graytemplate)  #sift匹配
        inname = inname + 1
    cv2.waitKey(0)
