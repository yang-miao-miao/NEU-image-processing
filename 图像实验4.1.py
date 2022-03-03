import cv2
import numpy as np
import cv2 as cv
current_pos = None
tl = None
br = None
im=cv.imread('./shiyan4/473.bmp')
#鼠标事件
def get_rect(im, title='get_rect'):   #   (a,b) = get_rect(im, title='get_rect')
    mouse_params = {'tl': None, 'br': None, 'current_pos': None,
        'released_once': False}
    cv2.namedWindow(title)
    cv2.moveWindow(title, 100, 100)
    def onMouse(event, x, y, flags, param):
        param['current_pos'] = (x, y)
        if param['tl'] is not None and not (flags & cv2.EVENT_FLAG_LBUTTON):
            param['released_once'] = True
        if flags & cv2.EVENT_FLAG_LBUTTON:
            if param['tl'] is None:
                param['tl'] = param['current_pos']
            elif param['released_once']:
                param['br'] = param['current_pos']
    cv2.setMouseCallback(title, onMouse, mouse_params)
    cv2.imshow(title, im)
    while mouse_params['br'] is None:
        im_draw = np.copy(im)
        if mouse_params['tl'] is not None:
            cv2.rectangle(im_draw, mouse_params['tl'],
                mouse_params['current_pos'], (255, 0, 0))
        cv2.imshow(title, im_draw)
        _ = cv2.waitKey(10)
    cv2.destroyWindow(title)
    tl = (min(mouse_params['tl'][0], mouse_params['br'][0]),
        min(mouse_params['tl'][1], mouse_params['br'][1]))
    br = (max(mouse_params['tl'][0], mouse_params['br'][0]),
        max(mouse_params['tl'][1], mouse_params['br'][1]))
    return (tl, br)  #tl=(y1,x1), br=(y2,x2)
tl,br=get_rect(im)
print(tl,br)
car=im[tl[1]:br[1],tl[0]:br[0]]
#cv.imwrite('E:/Desktop/car_model.jpg',car)
cv.imshow('car',car)
cv.waitKey(0)
cv.destroyAllWindows()