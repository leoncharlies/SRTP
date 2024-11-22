import numpy as np
import cv2
import matplotlib.pyplot as plt
import csv
import os
import shutil
import re

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

def canny_edge_detect(img_path):
    img=cv2.imread(img_path)
    low_threshold=50
    high_threshold=150
    edges=cv2.Canny(img,low_threshold,high_threshold)
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(img, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Canny Edge Detection")
    plt.imshow(edges, cmap="gray")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

def read_and_get_name(folder_path):
    file_data = []

    for filename in os.listdir(folder_path):
        if filename.endswith('.png'):
            pattern = r'([\d-]+)-(\([\d., ]+\))-(\([\d., ]+\))'
            match = re.match(pattern, filename)
            
            if match:
                id_part = match.group(1)
                righttop = match.group(2)
                leftbottom = match.group(3)

                file_data.append([id_part, righttop, leftbottom])

    output_file = '241119/output_list/img_info.csv'
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'righttop', 'leftbottom'])
        writer.writerows(file_data)

    print(f"数据已保存到 {output_file}")


read_and_get_name("origindata/2024-09-26 00-27")
#canny_edge_detect("241119/img_list/001_resize.png")
# resize_img("241119/img_list/001_reverse.png","241119/img_list/002.png")
# ignore_gray("241119/001.png")