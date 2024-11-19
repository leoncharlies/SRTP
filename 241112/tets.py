from PIL import Image
import numpy as np
import cv2
import pandas as pd

def extract_coordinates(point_str):
    """从 WKT 格式的 POINT 字符串中提取坐标"""
    coords = point_str.replace("POINT (", "").replace(")", "").split()
    return float(coords[0]), float(coords[1])

if __name__ == '__main__':
    data=pd.read_csv('data.csv')

