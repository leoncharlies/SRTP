import cv2
import numpy as np

# 读取两张图像
img1 = cv2.imread('241111/10-10-1-01.png', cv2.IMREAD_GRAYSCALE)  # 场景图像
img2 = cv2.imread('241111/10-10-1-02.png', cv2.IMREAD_GRAYSCALE)  # 包含道路的图像

# 检查图像是否成功加载
if img1 is None or img2 is None:
    print("图像加载失败")
    exit()

# 设置目标高度并缩放图像
target_height = 600
scale1 = target_height / img1.shape[0]
scale2 = target_height / img2.shape[0]
img1_resized = cv2.resize(img1, (int(img1.shape[1] * scale1), target_height), interpolation=cv2.INTER_AREA)
img2_resized = cv2.resize(img2, (int(img2.shape[1] * scale2), target_height), interpolation=cv2.INTER_AREA)

# 初始化 SIFT 检测器
sift = cv2.SIFT_create()

# 检测关键点并计算描述符
keypoints1, descriptors1 = sift.detectAndCompute(img1_resized, None)
keypoints2, descriptors2 = sift.detectAndCompute(img2_resized, None)

# 使用 BFMatcher 进行描述符匹配
bf = cv2.BFMatcher(cv2.NORM_L2)
matches = bf.knnMatch(descriptors2, descriptors1, k=2)

# 使用 Lowe's ratio test 筛选匹配点
good_matches = []
for m, n in matches:
    if m.distance < 0.9 * n.distance:
        good_matches.append(m)

# 如果有足够的匹配点
if len(good_matches) > 4:
    # 提取匹配点的坐标
    src_pts = np.float32([keypoints2[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints1[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # 计算单应性矩阵
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # 使用单应性矩阵进行透视变换，得到图像2在图像1中的位置
    h, w = img2_resized.shape
    pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
    dst_pts_transformed = cv2.perspectiveTransform(pts, M)

    # 在场景图像中绘制匹配区域
    img1_color = cv2.cvtColor(img1_resized, cv2.COLOR_GRAY2BGR)
    cv2.polylines(img1_color, [np.int32(dst_pts_transformed)], True, (0, 255, 0), 3, cv2.LINE_AA)

    # 绘制匹配点
    img_matches = cv2.drawMatches(img2_resized, keypoints2, img1_resized, keypoints1, good_matches, None,
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # 显示匹配结果
    cv2.imshow('Matches', img_matches)
    cv2.imshow('Detected Road in Scene', img1_color)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

else:
    print("匹配点不足，无法进行匹配")
