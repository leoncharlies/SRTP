import pandas as pd
import ast
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import os 
import numpy as np
import re

def draw_selected_road_by_from_node(map_csv, crop_csv, from_node, img_width=16000, img_height=16000, scale=100000, offset_x=104, offset_y=30.7):
    """
    根据 from_node 绘制地图上的某条道路，并找到包含该道路的裁剪区域。

    参数：
        map_csv (str): 包含地图线段数据的 CSV 文件路径。
        crop_csv (str): 包含裁剪区域信息的 CSV 文件路径。
        from_node (int): 要选择的目标道路的起始节点编号。
        img_width (int): 输出图片宽度，默认 16000。
        img_height (int): 输出图片高度，默认 16000。
        scale (float): 经纬度到像素的比例，默认 100000。
        offset_x (float): 地图的经度偏移量，默认 104。
        offset_y (float): 地图的纬度偏移量，默认 30.7。
    """
    # 读取CSV文件
    map_df = pd.read_csv(map_csv)
    crop_df = pd.read_csv(crop_csv)

    # 筛选出指定 from_node 的道路记录
    target_row = map_df[map_df['from_node'] == from_node]
    if target_row.empty:
        print(f"未找到 from_node = {from_node} 的道路")
        return None, None

    # 提取目标道路的 polyline 信息
    target_polyline = target_row.iloc[0]['polyline']
    # 解析 polyline 为坐标列表
    polyline_coords = [
        tuple(map(float, coord.split()))
        for coord in target_polyline.replace("LINESTRING (", "").replace(")", "").split(", ")
    ]

    img = Image.new('RGB', (img_width, img_height), 'white')
    draw = ImageDraw.Draw(img)

    # 转换坐标并绘制线段
    line_pixels = []
    for coord in polyline_coords:
        pixel = (
            int((coord[0] - offset_x) * scale),
            int((offset_y - coord[1]) * scale)
        )
        line_pixels.append(pixel)

    if len(line_pixels) > 1:
        draw.line(line_pixels, fill='black', width=2)

    target_crop_name = None
    cropped_img = None
    for index, row in crop_df.iterrows():
        crop_id = row['id']
        right_top = ast.literal_eval(row['righttop'])  # 将字符串转换为元组
        left_bottom = ast.literal_eval(row['leftbottom'])

        crop_lon_min = left_bottom[1]
        crop_lon_max = right_top[1]
        crop_lat_min = left_bottom[0]
        crop_lat_max = right_top[0]

        # 判断线段是否在裁剪区域范围内
        for coord in polyline_coords:
            if crop_lon_min <= coord[0] <= crop_lon_max and crop_lat_min <= coord[1] <= crop_lat_max:
                target_crop_name = crop_id

                # 计算裁剪区域的像素范围
                rect_top_left = (
                    int((crop_lon_min - offset_x) * scale),
                    int((offset_y - crop_lat_max) * scale)
                )
                rect_bottom_right = (
                    int((crop_lon_max - offset_x) * scale),
                    int((offset_y - crop_lat_min) * scale)
                )

                # 裁剪图片
                cropped_img = img.crop((*rect_top_left, *rect_bottom_right))
                break
        if target_crop_name:
            break

    return target_crop_name, cropped_img

def find_and_resize_images_with_regex(crop_name, cropped_image, folder_path):
    """
    使用正则表达式查找指定文件夹中的图片文件，并将其与裁剪后的图片进行大小对齐。

    参数：
        crop_name (str): 裁剪区域名称，用于匹配文件名（正则匹配前缀）。
        cropped_image (PIL.Image.Image): 裁剪后的图片。
        folder_path (str): 文件夹路径，用于搜索对应名称的图片。

    返回：
        tuple: 两张对齐大小的图片 (resized_target_image, resized_cropped_image)。
    """
    if not crop_name:
        print("裁剪区域名称为空，无法查找对应图片。")
        return None, None

    # 构建正则表达式，匹配以 crop_name 开头的文件名
    pattern = re.compile(rf"^{crop_name}.*\.(png|jpg|jpeg)$", re.IGNORECASE)

    # 搜索与裁剪区域名称匹配的图片文件
    target_image_path = None
    for file_name in os.listdir(folder_path):
        if pattern.match(file_name):
            target_image_path = os.path.join(folder_path, file_name)
            break

    if not target_image_path:
        print(f"未在文件夹 {folder_path} 中找到匹配的图片文件。")
        return None, None

    # 加载文件夹中的目标图片
    target_image = Image.open(target_image_path)

    # 调整大小
    cropped_size = cropped_image.size  # 获取裁剪图片的大小 (width, height)
    resized_target_image = target_image.resize(cropped_size, Image.LANCZOS)
    resized_cropped_image = cropped_image.resize(cropped_size, Image.LANCZOS)

    # 返回调整大小后的图片
    return resized_target_image, resized_cropped_image



def display_images(image1, image2, title1="Image 1", title2="Image 2"):
    """
    使用 Matplotlib 显示两张图片。

    参数：
        image1 (PIL.Image.Image): 第一张图片。
        image2 (PIL.Image.Image): 第二张图片。
        title1 (str): 第一张图片的标题。
        title2 (str): 第二张图片的标题。
    """
    if image1 is None or image2 is None:
        print("其中一张图片为空，无法显示。")
        return

    # 转换为 numpy 数组以供显示
    image1_array = np.array(image1)
    image2_array = np.array(image2)

    # 显示图片
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image1_array)
    plt.title(title1)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(image2_array)
    plt.title(title2)
    plt.axis('off')

    plt.tight_layout()
    plt.show()

def map_cropped_to_target_rgb(cropped_image, target_image):
    """
    查找裁剪图片的像素点对应到目标图片的像素点的 RGB 值。

    参数：
        cropped_image (PIL.Image.Image): 裁剪图片。
        target_image (PIL.Image.Image): 目标图片。

    返回：
        list: 每个像素对应的 RGB 值，格式为 [(x, y, (r, g, b)), ...]。
    """
    if cropped_image.size != target_image.size:
        raise ValueError("裁剪图片和目标图片的大小必须相同！")

    # 转换为 numpy 数组以便处理像素数据
    cropped_array = np.array(cropped_image)
    target_array = np.array(target_image)

    # 获取裁剪图片的非白色像素点位置
    non_white_pixels = []
    for y in range(cropped_array.shape[0]):  # 高度
        for x in range(cropped_array.shape[1]):  # 宽度
            if not np.array_equal(cropped_array[y, x], [255, 255, 255]):  # 非白色像素
                non_white_pixels.append((x, y))

    # 收集目标图片中对应像素的 RGB 值
    mapped_pixels = []
    for x, y in non_white_pixels:
        rgb_value = tuple(target_array[y, x])  # 提取对应像素的 RGB 值
        mapped_pixels.append((x, y, rgb_value))

    return mapped_pixels

def check_road_status(mapped_pixels, green_threshold=0.4):
    """
    根据绿色像素的比例判断道路是否通畅。

    参数：
        mapped_pixels (list): 由 (x, y, (r, g, b)) 组成的列表，表示裁剪图像中非白色像素点及其对应的 RGB 值。
        green_threshold (float): 绿色像素占比的阈值，默认为 0.4（即 40%）。
        
    返回：
        str: 返回道路的状态（'通畅' 或 '不通畅'）。
    """
    # 定义绿色的 RGB 范围
    def is_green(rgb):
        r, g, b = rgb
        return r < 150 and g > 100 and b < 150

    # 计算绿色像素的数量
    green_count = 0
    total_count = len(mapped_pixels)

    for _, _, rgb in mapped_pixels:
        if is_green(rgb):
            green_count += 1

    # 计算绿色像素占比
    green_ratio = green_count / total_count if total_count > 0 else 0
    print(f"Green ratio: {green_ratio}")
    print(f"Green count: {green_count}")
    print(f"Total count: {total_count}")
    # 判断道路是否通畅
    if green_ratio >= green_threshold:
        return '通畅'
    else:
        return '不通畅'

if __name__ == '__main__':
    map_csv = "datas/origindata/2024-02-03 11-10_network.csv"
    crop_csv = "datas/exp/output_list/img_info.csv"
    images_folder = "datas/origindata/2024-09-26 00-27"
    from_node = 3 #选择一条道路
    target_name,ttimg=draw_selected_road_by_from_node(map_csv,crop_csv,from_node)
    resized_target_image ,resized_cropped_image = find_and_resize_images_with_regex(target_name,ttimg,images_folder)
    #display_images(resized_target_image,resized_cropped_image)
    mapped_pixels = map_cropped_to_target_rgb(resized_cropped_image,resized_target_image)

    for pixel in mapped_pixels[:10]:
        print(f"Pixel at ({pixel[0]}, {pixel[1]}) in cropped image -> RGB in target image: {pixel[2]}")
    print(f"Road {from_node} status: {check_road_status(mapped_pixels)}")
