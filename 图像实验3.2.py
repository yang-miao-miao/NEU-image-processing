import cv2 as cv
img=cv.imread('./shiyan3/ab_thresh.jpg',0)
rows,cols=img.shape
a=img[:,:int(cols/2)]
b=img[:,int(cols/2)+50:]
cv.imshow('a',a)
cv.imshow('b',b)
cv.waitKey(0)
cv.destroyAllWindows()