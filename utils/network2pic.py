import pandas as pd
from PIL import Image, ImageDraw
from tqdm import tqdm
import ast
 
def extract_coordinates(point_str):
    """从 WKT 格式的 POINT 字符串中提取坐标"""
    coords = point_str.replace("POINT (", "").replace(")", "").split()
    return float(coords[0]), float(coords[1])

def map_coordinates(lon, lat, lon_min, lon_max, lat_min, lat_max, img_width, img_height):
    """将经纬度映射到图像像素坐标"""
    x = int((lon - lon_min) / (lon_max - lon_min) * img_width)
    y = int((lat_max - lat) / (lat_max - lat_min) * img_height)
    return x, y

def draw_full_image(csv_file, img_width=16000, img_height=16000, scale=100000, offset_x=104, offset_y=30.7):
    # 读取 CSV 文件
    df = pd.read_csv(csv_file)
    
    img = Image.new('RGB', (img_width, img_height), 'white')
    draw = ImageDraw.Draw(img)

    # 使用 tqdm 显示进度条
    for index, row in tqdm(df.iterrows(), total=len(df), desc="绘制线段"):
        # 提取起点和终点的坐标
        start_point = extract_coordinates(row['start_point'])
        end_point = extract_coordinates(row['end_point'])

        # 将经纬度转换为图像坐标
        start_pixel = (int((start_point[0] - offset_x) * scale), int((offset_y - start_point[1]) * scale))
        end_pixel = (int((end_point[0] - offset_x) * scale), int((offset_y - end_point[1]) * scale))

        # 确保点在图像范围内
        if 0 <= start_pixel[0] < img_width and 0 <= start_pixel[1] < img_height and \
           0 <= end_pixel[0] < img_width and 0 <= end_pixel[1] < img_height:
            # 绘制线段
            draw.line([start_pixel, end_pixel], fill='black', width=2)

    return img

def draw_img_cropped(csv_file, crop_lon_min, crop_lon_max, crop_lat_min, crop_lat_max,img_width=16000, img_height=16000, scale=100000, offset_x=104, offset_y=30.7):
    df = pd.read_csv(csv_file)
    
    img = Image.new('RGB', (img_width, img_height), 'white')
    draw = ImageDraw.Draw(img)

    for index, row in tqdm(df.iterrows(), total=len(df), desc="绘制线段"):

        start_point = extract_coordinates(row['start_point'])
        end_point = extract_coordinates(row['end_point'])

        start_pixel = (int((start_point[0] - offset_x) * scale), int((offset_y - start_point[1]) * scale))
        end_pixel = (int((end_point[0] - offset_x) * scale), int((offset_y - end_point[1]) * scale))

        if 0 <= start_pixel[0] < img_width and 0 <= start_pixel[1] < img_height and \
           0 <= end_pixel[0] < img_width and 0 <= end_pixel[1] < img_height:

            draw.line([start_pixel, end_pixel], fill='black', width=2)

    img = img.crop((int((crop_lon_min - offset_x) * scale), int((offset_y - crop_lat_max) * scale),
                    int((crop_lon_max - offset_x) * scale), int((offset_y - crop_lat_min) * scale)))
    return img

def draw_single_road(csv_file, road_id, crop_lon_min, crop_lon_max, crop_lat_min, crop_lat_max, 
                     img_width=16000, img_height=16000, scale=100000, offset_x=104, offset_y=30.7):
    """
    绘制单条道路，同时保持整体地图的比例，并裁剪到指定区域。
    
    :param csv_file: 包含道路数据的 CSV 文件
    :param road_id: 指定的道路 ID，用于过滤只绘制该道路
    :param crop_lon_min: 裁剪区域的最小经度
    :param crop_lon_max: 裁剪区域的最大经度
    :param crop_lat_min: 裁剪区域的最小纬度
    :param crop_lat_max: 裁剪区域的最大纬度
    :param img_width: 图片宽度
    :param img_height: 图片高度
    :param scale: 经纬度到像素的比例
    :param offset_x: X 轴偏移量
    :param offset_y: Y 轴偏移量
    :return: 裁剪后的图像
    """
    df = pd.read_csv(csv_file)
    
    # 过滤数据，只保留指定道路
    road_df = df[df['road_id'] == road_id]  # 假设 'road_id' 是道路的唯一标识字段
    
    # 创建白色背景图片
    img = Image.new('RGB', (img_width, img_height), 'white')
    draw = ImageDraw.Draw(img)

    # 绘制指定道路
    for index, row in tqdm(road_df.iterrows(), total=len(road_df), desc="绘制线段"):
        start_point = extract_coordinates(row['start_point'])
        end_point = extract_coordinates(row['end_point'])

        start_pixel = (int((start_point[0] - offset_x) * scale), int((offset_y - start_point[1]) * scale))
        end_pixel = (int((end_point[0] - offset_x) * scale), int((offset_y - end_point[1]) * scale))

        if 0 <= start_pixel[0] < img_width and 0 <= start_pixel[1] < img_height and \
           0 <= end_pixel[0] < img_width and 0 <= end_pixel[1] < img_height:

            draw.line([start_pixel, end_pixel], fill='black', width=2)

    # 裁剪到指定范围
    #img = img.crop((int((crop_lon_min - offset_x) * scale), int((offset_y - crop_lat_max) * scale),
                    #int((crop_lon_max - offset_x) * scale), int((offset_y - crop_lat_min) * scale)))
    
    return img

def draw_img_with_crops(map_csv, crop_csv, img_width=16000, img_height=16000, scale=100000, offset_x=104, offset_y=30.7):
    """
    绘制地图，并在地图上标注多个裁剪区域。

    参数：
        map_csv (str): 包含地图线段数据的 CSV 文件路径。
        crop_csv (str): 包含裁剪区域信息的 CSV 文件路径。
        img_width (int): 输出图片宽度，默认 16000。
        img_height (int): 输出图片高度，默认 16000。
        scale (float): 经纬度到像素的比例，默认 100000。
        offset_x (float): 地图的经度偏移量，默认 104。
        offset_y (float): 地图的纬度偏移量，默认 30.7。
    """

    map_df = pd.read_csv(map_csv)
    crop_df = pd.read_csv(crop_csv)

    img = Image.new('RGB', (img_width, img_height), 'white')
    draw = ImageDraw.Draw(img)

    for index, row in tqdm(map_df.iterrows(), total=len(map_df), desc="绘制线段"):
        start_point = extract_coordinates(row['start_point'])
        end_point = extract_coordinates(row['end_point'])

        start_pixel = (
            int((start_point[0] - offset_x) * scale),
            int((offset_y - start_point[1]) * scale)
        )
        end_pixel = (
            int((end_point[0] - offset_x) * scale),
            int((offset_y - end_point[1]) * scale)
        )

        if 0 <= start_pixel[0] < img_width and 0 <= start_pixel[1] < img_height and \
           0 <= end_pixel[0] < img_width and 0 <= end_pixel[1] < img_height:
            draw.line([start_pixel, end_pixel], fill='black', width=2)

    for index, row in crop_df.iterrows():
        crop_id = row['id']
        right_top = ast.literal_eval(row['righttop'])  # 将字符串转换为元组
        left_bottom = ast.literal_eval(row['leftbottom'])

        crop_lon_min = left_bottom[1]
        crop_lon_max = right_top[1]
        crop_lat_min = left_bottom[0]
        crop_lat_max = right_top[0]

        if crop_lon_min < offset_x or crop_lon_max < offset_x or crop_lat_min > offset_y or crop_lat_max > offset_y:
            continue

        rect_top_left = (
            int((crop_lon_min - offset_x) * scale),
            int((offset_y - crop_lat_max) * scale)
        )
        rect_bottom_right = (
            int((crop_lon_max - offset_x) * scale),
            int((offset_y - crop_lat_min) * scale)
        )

        draw.rectangle([rect_top_left, rect_bottom_right], outline='red', width=3)
        draw.text(rect_top_left, str(crop_id), fill='blue')

    return img

def draw_img_with_one_line(map_csv, crop_csv, target_start, target_end, img_width=16000, img_height=16000, scale=100000, offset_x=104, offset_y=30.7):
    """
    绘制地图，并只绘制指定的一条线，同时找到包含该线的裁剪区域。

    参数：
        map_csv (str): 包含地图线段数据的 CSV 文件路径。
        crop_csv (str): 包含裁剪区域信息的 CSV 文件路径。
        target_start (tuple): 目标线段的起点 (纬度, 经度)。
        target_end (tuple): 目标线段的终点 (纬度, 经度)。
        img_width (int): 输出图片宽度，默认 16000。
        img_height (int): 输出图片高度，默认 16000。
        scale (float): 经纬度到像素的比例，默认 100000。
        offset_x (float): 地图的经度偏移量，默认 104。
        offset_y (float): 地图的纬度偏移量，默认 30.7。
    """

    map_df = pd.read_csv(map_csv)
    crop_df = pd.read_csv(crop_csv)

    img = Image.new('RGB', (img_width, img_height), 'white')
    draw = ImageDraw.Draw(img)

    # 计算目标线段的像素坐标
    target_start_pixel = (
        int((target_start[1] - offset_x) * scale),
        int((offset_y - target_start[0]) * scale)
    )
    target_end_pixel = (
        int((target_end[1] - offset_x) * scale),
        int((offset_y - target_end[0]) * scale)
    )

    # 绘制指定线段
    if 0 <= target_start_pixel[0] < img_width and 0 <= target_start_pixel[1] < img_height and \
       0 <= target_end_pixel[0] < img_width and 0 <= target_end_pixel[1] < img_height:
        draw.line([target_start_pixel, target_end_pixel], fill='black', width=2)

    # 找到包含线段的裁剪区域
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
        if (crop_lon_min <= target_start[1] <= crop_lon_max and crop_lat_min <= target_start[0] <= crop_lat_max) or \
           (crop_lon_min <= target_end[1] <= crop_lon_max and crop_lat_min <= target_end[0] <= crop_lat_max):
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

    return target_crop_name, cropped_img



#7-12-1-(30.676134432537427, 104.07512870773678)-(30.666020568780848, 104.06183031544094)
if __name__ == '__main__':
    #csv_file_path = 'origindata/2024-02-03 11-10_network.csv'  
    output_image_path = 'output/map_with_crops.png'  
    target_crop_name,cropped_img=draw_img_with_one_line('origindata/2024-02-03 11-10_network.csv','origindata/2024-02-03 11-10_crops.csv',(30.676134432537427, 104.07512870773678),(30.666020568780848, 104.06183031544094))
    img.save(output_image_path, format='PNG', optimize=True)
    print(f"图像已保存到 {output_image_path}")