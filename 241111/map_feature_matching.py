import cv2 
import numpy as np
from cv2 import ximgproc

def detect_intersections(skeleton):
    # 使用卷积核检测交叉点
    kernel = np.array([[1,1,1],
                      [1,10,1],
                      [1,1,1]])
    crossing_points = cv2.filter2D(skeleton.astype(float), -1, kernel)
    # 找出值大于11的点（表示交叉点）
    intersections = np.where(crossing_points > 11)
    return zip(intersections[1], intersections[0])  # x,y坐标

# 为每个交叉点创建描述符
# 为每个交叉点创建描述符
def create_descriptor(skeleton, point, radius=10):
    x, y = point
    h, w = skeleton.shape
    # 保证在图像边界内
    if y - radius >= 0 and y + radius < h and x - radius >= 0 and x + radius < w:
        region = skeleton[y-radius:y+radius, x-radius:x+radius]
        return region.flatten()  # 确保返回一致的大小
    else:
        return None  # 返回None表示跳过边界点


def skeletonize(img):
    size = np.size(img)
    skel = np.zeros(img.shape, np.uint8)
    
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    done = False
    
    while not done:
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()
        
        zeros = size - cv2.countNonZero(img)
        if zeros == size:
            done = True
            
    return skel

def match_descriptors(desc1, desc2):
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)  # 使用BFMatcher进行特征匹配
    matches = bf.match(np.array(desc1), np.array(desc2))
    matches = sorted(matches, key=lambda x: x.distance)  # 按距离排序
    return matches  # 返回匹配结果



# 使用RANSAC算法验证匹配的正确性
def geometric_verification(matches, points1, points2):
    src_pts = np.float32([points1[m[0]] for m in matches])
    dst_pts = np.float32([points2[m[1]] for m in matches])
    
    # 计算变换矩阵
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
    return H, mask


# 应用变换矩阵
def apply_transform(img1, img2, H):
    h, w = img2.shape[:2]
    transformed = cv2.warpPerspective(img1, H, (w, h))
    return transformed

# 可视化结果
def visualize_matching(img1, img2, transformed):
    # 创建叠加显示
    overlay = cv2.addWeighted(img2, 0.5, transformed, 0.5, 0)
    return overlay

def resizeimg(img,scale):
    dn=(int(img.shape[1]*scale),int(img.shape[0]*scale))
    img1=cv2.resize(img,dn,interpolation=cv2.INTER_AREA)
    return img1

def main():
    # 1. 读取图像
    img1 = cv2.imread('10-10-1-01.png')  # 你的第一张图片（绿色线条的地图）
    img2 = cv2.imread('10-10-1-02.png')  # 你的第二张图片（黑白线条图）
    
    img1=resizeimg(img1,0.5)
    img2=resizeimg(img2,0.5)

    if img1 is None or img2 is None:
        print("Error: Could not read images")
        return

    green_channel = img1[:,:,1]
    _, binary1 = cv2.threshold(green_channel, 127, 255, cv2.THRESH_BINARY)
    _, binary2 = cv2.threshold(img2, 127, 255, cv2.THRESH_BINARY)

    # 确保图像是单通道
    if len(binary1.shape) > 2:
        binary1 = cv2.cvtColor(binary1, cv2.COLOR_BGR2GRAY)
    if len(binary2.shape) > 2:
        binary2 = cv2.cvtColor(binary2, cv2.COLOR_BGR2GRAY)

    skeleton1 = skeletonize(binary1)
    skeleton2 = skeletonize(binary2)

    intersections1 = list(detect_intersections(skeleton1))
    intersections2 = list(detect_intersections(skeleton2))

    descriptors1 = [d for d in (create_descriptor(skeleton1, p) for p in intersections1) if d is not None]
    descriptors2 = [d for d in (create_descriptor(skeleton2, p) for p in intersections2) if d is not None]


    matches = match_descriptors(descriptors1, descriptors2)

    H, mask = geometric_verification(matches, intersections1, intersections2)

    transformed = apply_transform(img1, img2, H)

    result = visualize_matching(img1, img2, transformed)
    
    # 10. 显示结果
    cv2.imshow('Matching Result', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # 11. 保存结果
    cv2.imwrite('matching_result.png', result)

if __name__ == "__main__":
    main()