import cv2 as cv
import matplotlib.pyplot as plt
ab=cv.imread('./shiyan3/ab.jpg',0)
cv.imshow('img',ab)
hist = cv.calcHist([ab], [0], None, [256], [0, 256])
plt.plot(hist)
plt.show()
ret,thresh=cv.threshold(ab,100,255,cv.THRESH_BINARY)
cv.imshow('thresh',thresh)
#cv.imwrite('./ab_thresh.jpg',thresh)
cv.waitKey(0)
cv.destroyAllWindows()