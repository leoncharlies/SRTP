import cv2
import numpy as np

# 读取两张图像
img1 = cv2.imread('../241111/10-10-1-01.png', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('../241111/10-10-1-02.png', cv2.IMREAD_GRAYSCALE)

# 检查图像是否成功加载
if img1 is None or img2 is None:
    print("图像加载失败")
    exit()
target_height = 600  # 可以根据需要设置目标高度
scale1 = target_height / img1.shape[0]
scale2 = target_height / img2.shape[0]
img1_resized = cv2.resize(img1, (int(img1.shape[1] * scale1), target_height), interpolation=cv2.INTER_AREA)
img2_resized = cv2.resize(img2, (int(img2.shape[1] * scale2), target_height), interpolation=cv2.INTER_AREA)

sift = cv2.SIFT_create()

# 检测关键点并计算描述符
keypoints1, descriptors1 = sift.detectAndCompute(img1_resized, None)
keypoints2, descriptors2 = sift.detectAndCompute(img2_resized, None)

# 使用 BFMatcher 进行描述符匹配
bf = cv2.BFMatcher(cv2.NORM_L2)
matches = bf.match(descriptors1, descriptors2)

# 根据距离排序匹配点，距离越小越好
matches = sorted(matches, key=lambda x: x.distance)

num_good_matches = 100 #选择前50个匹配点

good_matches = matches[:num_good_matches]

img_matches = cv2.drawMatches(img1_resized, keypoints1, img2_resized, keypoints2, good_matches, None,
                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
scale_percent = 100 # 缩放百分比
width = int(img_matches.shape[1] * scale_percent / 100)
height = int(img_matches.shape[0] * scale_percent / 100)
dim = (width, height)

img_matches = cv2.resize(img_matches, dim, interpolation=cv2.INTER_AREA)
# 显示结果
cv2.imshow("SIFT Feature Matching", img_matches)

cv2.waitKey(0)
cv2.destroyAllWindows()