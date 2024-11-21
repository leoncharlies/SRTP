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

def resize_img(img01_path,img02_path):
    img01=cv2.imread(img01_path)
    img02=cv2.imread(img02_path)
    height1, width1 = img01.shape[:2]
    height2, width2 = img02.shape[:2]
    target_size=(max(width1,width2),max(height1,height2))
    img01=cv2.resize(img01,target_size,interpolation=cv2.INTER_AREA)
    img02=cv2.resize(img02,target_size,interpolation=cv2.INTER_AREA)
    cv2.imwrite("241119/img_list/001_resize.png",img01)
    cv2.imwrite("241119/img_list/002_resize.png",img02)

def roberts_edge_detect(img_path):
    pass



# resize_img("241119/img_list/001_reverse.png","241119/img_list/002.png")
# ignore_gray("241119/001.png")