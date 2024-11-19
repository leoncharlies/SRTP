import pandas as pd
from PIL import Image, ImageDraw
from tqdm import tqdm

def extract_coordinates(point_str):
    """从 WKT 格式的 POINT 字符串中提取坐标"""
    coords = point_str.replace("POINT (", "").replace(")", "").split()
    return float(coords[0]), float(coords[1])

def map_coordinates(lon, lat, lon_min, lon_max, lat_min, lat_max, img_width, img_height):
    """将经纬度映射到图像像素坐标"""
    x = int((lon - lon_min) / (lon_max - lon_min) * img_width)
    y = int((lat_max - lat) / (lat_max - lat_min) * img_height)
    return x, y

def draw_lines_from_csv(csv_file, output_image, img_width=32000, img_height=32000):
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

    # 创建高分辨率图像
    img = Image.new('RGB', (img_width, img_height), 'white')
    draw = ImageDraw.Draw(img)

    # 使用 tqdm 显示进度条
    for index, row in tqdm(df.iterrows(), total=len(df), desc="绘制线段"):
        # 提取起点和终点的坐标
        start_point = extract_coordinates(row['start_point'])
        end_point = extract_coordinates(row['end_point'])

        # 将经纬度转换为图像坐标
        start_pixel = map_coordinates(start_point[0], start_point[1], lon_min, lon_max, lat_min, lat_max, img_width, img_height)
        end_pixel = map_coordinates(end_point[0], end_point[1], lon_min, lon_max, lat_min, lat_max, img_width, img_height)

        # 确保点在图像范围内
        if 0 <= start_pixel[0] < img_width and 0 <= start_pixel[1] < img_height and \
           0 <= end_pixel[0] < img_width and 0 <= end_pixel[1] < img_height:
            # 绘制线段
            draw.line([start_pixel, end_pixel], fill='black', width=4)  # 设置较粗的线条

    # 缩小图像以优化质量
    img = img.resize((img_width // 2, img_height // 2), Image.LANCZOS)

    # 保存生成的图像
    img.save(output_image, format='PNG', optimize=True)
    print(f"图像已保存到 {output_image}")

if __name__ == '__main__':
    csv_file_path = 'origindata/2024-02-03 11-10_network.csv'  # 请根据需要更改 CSV 文件路径
    output_image_path = 'utils/output_image.png'  # 输出图像文件名
    draw_lines_from_csv(csv_file_path, output_image_path)
