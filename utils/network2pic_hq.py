import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import PIL
from tqdm import tqdm
import random
PIL.Image.MAX_IMAGE_PIXELS = 1000000000  # 解决 PIL.Image.DecompressionBombError: Image size (1000000000 pixels) exceeds limit of 100000000 pixels

def extract_coordinates(point_str):
    """从 WKT 格式的 POINT 字符串中提取坐标"""
    coords = point_str.replace("POINT (", "").replace(")", "").split()
    return float(coords[0]), float(coords[1])


def map_coordinates(lon, lat, lon_min, lon_max, lat_min, lat_max, img_width, img_height):
    """将经纬度映射到图像像素坐标"""
    x = int((lon - lon_min) / (lon_max - lon_min) * img_width)
    y = int((lat_max - lat) / (lat_max - lat_min) * img_height)
    return x, y


def calculate_crop_pixel(lon, lat, reference_lon, reference_lat, reference_pixel, lon_min, lon_max, lat_min, lat_max, img_width, img_height):
    """
    根据基准点计算裁剪区域的像素坐标
    """
    # 基准点的像素坐标
    ref_x, ref_y = reference_pixel

    # 基准点相对的经纬度偏移
    x_offset = (lon - reference_lon) / (lon_max - lon_min) * img_width
    y_offset = (reference_lat - lat) / (lat_max - lat_min) * img_height

    # 计算裁剪点的像素坐标
    x = int(ref_x + x_offset)
    y = int(ref_y + y_offset)
    return x, y


def draw_and_crop_image(csv_file, crop_lon_min, crop_lon_max, crop_lat_min, crop_lat_max, img_width=32000, img_height=32000):
    """
    在生成图像时记录基准点，并直接裁剪图片。

    :param csv_file: 输入的CSV文件路径
    :param crop_lon_min: 裁剪区域的最小经度
    :param crop_lon_max: 裁剪区域的最大经度
    :param crop_lat_min: 裁剪区域的最小纬度
    :param crop_lat_max: 裁剪区域的最大纬度
    :param img_width: 原始图像宽度（像素）
    :param img_height: 原始图像高度（像素）
    """
    # 读取 CSV 文件
    df = pd.read_csv(csv_file)

    # 动态获取经纬度范围
    lon_min = min(df['start_point'].apply(lambda p: extract_coordinates(p)[0]).min(),
                  df['end_point'].apply(lambda p: extract_coordinates(p)[0]).min())
    lon_max = max(df['start_point'].apply(lambda p: extract_coordinates(p)[0]).max(),
                  df['end_point'].apply(lambda p: extract_coordinates(p)[0]).max())
    lat_min = min(df['start_point'].apply(lambda p: extract_coordinates(p)[1]).min(),
                  df['end_point'].apply(lambda p: extract_coordinates(p)[1]).min())
    lat_max = max(df['start_point'].apply(lambda p: extract_coordinates(p)[1]).max(),
                  df['end_point'].apply(lambda p: extract_coordinates(p)[1]).max())

    # 验证裁剪经纬度范围
    if crop_lon_min >= crop_lon_max or crop_lat_min >= crop_lat_max:
        raise ValueError("裁剪区域的经纬度范围无效：确保 crop_lon_min < crop_lon_max 且 crop_lat_min < crop_lat_max")

    # 创建高分辨率图像
    img = Image.new('RGB', (img_width, img_height), 'white')
    draw = ImageDraw.Draw(img)

    # 基准点经纬度
    reference_lon, reference_lat = lon_min, lat_max
    reference_pixel = map_coordinates(
        reference_lon, reference_lat,
        lon_min, lon_max, lat_min, lat_max,
        img_width, img_height
    )
    print(f"基准点经纬度: ({reference_lon}, {reference_lat})")
    print(f"基准点像素坐标: {reference_pixel}")

    # 绘制所有线段
    for index, row in tqdm(df.iterrows(), total=len(df), desc="绘制线段"):
        start_point = extract_coordinates(row['start_point'])
        end_point = extract_coordinates(row['end_point'])

        start_pixel = map_coordinates(start_point[0], start_point[1], lon_min, lon_max, lat_min, lat_max, img_width, img_height)
        end_pixel = map_coordinates(end_point[0], end_point[1], lon_min, lon_max, lat_min, lat_max, img_width, img_height)

        draw.line([start_pixel, end_pixel], fill='black', width=4)

    # 计算裁剪区域的像素坐标
    top_left_pixel = calculate_crop_pixel(crop_lon_min, crop_lat_max, reference_lon, reference_lat, reference_pixel, lon_min, lon_max, lat_min, lat_max, img_width, img_height)
    bottom_right_pixel = calculate_crop_pixel(crop_lon_max, crop_lat_min, reference_lon, reference_lat, reference_pixel, lon_min, lon_max, lat_min, lat_max, img_width, img_height)

    # 确保像素坐标范围有效
    left, right = max(0, top_left_pixel[0]), min(img_width, bottom_right_pixel[0])
    top, bottom = max(0, top_left_pixel[1]), min(img_height, bottom_right_pixel[1])

    if left >= right or top >= bottom:
        raise ValueError("裁剪区域像素坐标无效：确保裁剪范围不为空")

    print(f"裁剪区域像素坐标: left={left}, right={right}, top={top}, bottom={bottom}")

    # 裁剪图像
    cropped_img = img.crop((left, top, right, bottom))

    # 缩小裁剪后的图像
    cropped_img = cropped_img.resize((cropped_img.width // 2, cropped_img.height // 2), Image.LANCZOS)

    return cropped_img


# 测试代码
if __name__ == "__main__":
    cropped_image = draw_and_crop_image(
        csv_file="origindata/2024-02-03 11-10_network.csv",  # 替换为你的CSV文件路径
        crop_lon_min=103.91554800018673,
        crop_lon_max=103.92884639248257,
        crop_lat_min=30.605315160414357,
        crop_lat_max=30.61543537338345
    )
    cropped_image.save("/tmp/cropped_image.png")
    print("裁剪后的图像已保存为 cropped_image.png")
