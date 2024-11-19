import numpy as np
import cv2

def ignore_gray(img_path):

    img=cv2.imread(img_path)
    img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img_reverse=cv2.bitwise_not(img_gray)
    ret,img_binary=cv2.threshold(img_reverse,150,255,cv2.THRESH_BINARY)
    cv2.namedWindow("001",cv2.WINDOW_NORMAL)
    cv2.imshow("001",img_binary)
    cv2.imwrite("241119/001_reverse.png",img_binary)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


ignore_gray("241119/001.png")