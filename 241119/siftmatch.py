import cv2
import numpy as np


def sift_match_only_with_resize(image1_path, image2_path, output_path, max_matches=50):
    img001=cv2.imread(image1_path)
    img002=cv2.imread(image2_path)
    height1, width1 = img001.shape[:2]
    height2, width2 = img002.shape[:2]
    target_size =(max(height1, height2),max(width1, width2))
    img001=cv2.resize(img001,target_size,interpolation=cv2.INTER_AREA)
    img002=cv2.resize(img002,target_size,interpolation=cv2.INTER_AREA)
    
    if img001 is None or img002 is None:
        print("无法读取图像，请检查路径。")
        return
    gray001=cv2.cvtColor(img001,cv2.COLOR_BGR2GRAY)
    gray002=cv2.cvtColor(img002,cv2.COLOR_BGR2GRAY)
    sift=cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(img001, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img002, None)

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)

    matches = sorted(matches, key=lambda x: x.distance)

    result_img = cv2.drawMatches(
        img001, keypoints1, img002, keypoints2, 
        matches[:max_matches], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    cv2.imwrite(output_path, result_img)
    print(f"匹配结果已保存到 {output_path}")
    cv2.imshow('SIFT Matching', result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

def sift_match_ignore_gray(image1_path, image2_path, output_path='sift_matching_result.jpg', max_matches=50):
    """
    使用 SIFT 进行图像匹配，忽略图像001中的灰色道路线，仅匹配其他颜色的道路线。
    
    :param image1_path: 图像001的路径
    :param image2_path: 图像002的路径
    :param output_path: 匹配结果保存路径
    :param max_matches: 显示的最大匹配数量
    """
    # 读取图像
    img1 = cv2.imread(image1_path)  # 图像001
    img2 = cv2.imread(image2_path)  # 图像002

    if img1 is None or img2 is None:
        print("无法读取图像，请检查路径。")
        return

    # 将图像001转换为 HSV 颜色空间
    hsv1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)

    # 定义灰色的颜色范围（低饱和度区域）
    lower_gray = np.array([0, 0, 50])    # 最低灰色范围 (H, S, V)
    upper_gray = np.array([180, 50, 200])  # 最高灰色范围 (H, S, V)

    # 创建掩膜，识别灰色区域
    gray_mask = cv2.inRange(hsv1, lower_gray, upper_gray)

    # 反转掩膜，保留非灰色部分
    non_gray_mask = cv2.bitwise_not(gray_mask)

    # 应用掩膜，将灰色区域设置为黑色
    img1_no_gray = cv2.bitwise_and(img1, img1, mask=non_gray_mask)

    # 灰度化和反转颜色
    gray1 = cv2.cvtColor(img1_no_gray, cv2.COLOR_BGR2GRAY)
    inverted_gray1 = cv2.bitwise_not(gray1)  # 反转颜色

    # 将图像002灰度化
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # 初始化 SIFT 算法
    sift = cv2.SIFT_create()

    # 提取 SIFT 特征点和描述子
    keypoints1, descriptors1 = sift.detectAndCompute(inverted_gray1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

    # 使用 BFMatcher 进行特征匹配
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)

    # 根据距离排序匹配结果
    matches = sorted(matches, key=lambda x: x.distance)

    # 绘制匹配结果
    result_img = cv2.drawMatches(
        inverted_gray1, keypoints1, gray2, keypoints2, 
        matches[:max_matches], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    # 保存结果
    cv2.imwrite(output_path, result_img)
    print(f"匹配结果已保存到 {output_path}")

    # 显示结果
    cv2.imshow('SIFT Matching', result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

sift_match_only_with_resize('241119/001_reverse.png', '241119/002.png',output_path='241119/sift_match.png')
