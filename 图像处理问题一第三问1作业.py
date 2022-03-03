import numpy as np
import cv2 as cv
img=cv.imread('./images/1/Circuit_noise.jpg')
#中值滤波
img1=cv.medianBlur(img,5)
#NL-Means降噪
img2=cv.fastNlMeansDenoisingColored(img,None,10,10,7,21)
cv.imwrite('./imgwork/learn/que3/Circuit_noise/img1.jpg',img1)
cv.imwrite('./imgwork/learn/que3/Circuit_noise/img2.jpg',img2)
cv.imshow('img',img)
cv.imshow('a',img1)
cv.imshow('b',img2)
cv.waitKey(0)
cv.destroyAllWindows()