import cv2
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
for i in range(10):
    img = cv2.imread(('./shiyan4/pic/'+str(473+i)+'.bmp'))
    template = cv2.imread('./shiyan4/temp.bmp')
    cv.imshow('img', img)
    h, w = template.shape[:2]  # rows->h, cols->w
    # 相关系数匹配方法: cv2.TM_CCOEFF
    res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    left_top = max_loc  # 左上角
    right_bottom = (left_top[0] + w, left_top[1] + h)  # 右下角
    cv2.rectangle(img, left_top, right_bottom, 255, 2)  # 画出矩形位置
    car = img[left_top[1]:right_bottom[1], left_top[0]:right_bottom[0]]
    cv.imshow(('car'+str(i)), car)
    #cv.imwrite('car',car)
cv.waitKey(0)
cv.destroyAllWindows()