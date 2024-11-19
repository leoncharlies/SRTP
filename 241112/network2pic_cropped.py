import pandas as pd
from PIL import Image, ImageDraw
from tqdm import tqdm

def extract_coordinates(point_str):
    """从 WKT 格式的 POINT 字符串中提取坐标"""
    coords = point_str.replace("POINT (", "").replace(")", "").split()
    return float(coords[0]), float(coords[1])

def draw_full_image(csv_file, img_width=16000, img_height=16000):
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
        start_pixel = (int((start_point[0] - 104) * 100000), int((30.7 - start_point[1]) * 100000))
        end_pixel = (int((end_point[0] - 104) * 100000), int((30.7 - end_point[1]) * 100000))

        # 确保点在图像范围内
        if 0 <= start_pixel[0] < img_width and 0 <= start_pixel[1] < img_height and \
           0 <= end_pixel[0] < img_width and 0 <= end_pixel[1] < img_height:
            # 绘制线段
            draw.line([start_pixel, end_pixel], fill='black', width=2)

    return img

def crop_image_by_coordinates(img, top_right, bottom_left, img_width=16000, img_height=16000):
    """根据给定的经纬度范围对图像进行裁剪"""
    # 定义地理坐标到像素坐标的缩放因子和偏移量
    scale_x, scale_y = 100000, 100000
    offset_x, offset_y = 104, 30.7
    

    # 将经纬度坐标转换为像素坐标
    top_right_pixel = (int((top_right[1] - offset_x) * scale_x), int((offset_y - top_right[0]) * scale_y))
    bottom_left_pixel = (int((bottom_left[1] - offset_x) * scale_x), int((offset_y - bottom_left[0]) * scale_y))
    # 输出裁剪坐标以检查是否在范围内
    print("Top-right pixel:", top_right_pixel)
    print("Bottom-left pixel:", bottom_left_pixel)

    # 裁剪图像
    cropped_img = img.crop((bottom_left_pixel[0], top_right_pixel[1], top_right_pixel[0], bottom_left_pixel[1]))
    return cropped_img

if __name__ == '__main__':
    csv_file_path = 'origindata/2024-02-03 11-10_network.csv'  # CSV 文件路径
    output_image_path = '241112/output_image.png'  # 完整图片输出路径
    cropped_output_path = '241112/cropped_image.png'  # 截取后的图片路径
    
    # 绘制完整的图片
    img = draw_full_image(csv_file_path)
    img.save(output_image_path)
    print(f"完整图像已保存到 {output_image_path}")

    # 截取指定范围的图片
    top_right_coords = (30.61543537338345, 103.92884639248257)
    bottom_left_coords = (30.605315160414357, 103.91554800018673)
    cropped_img = crop_image_by_coordinates(img, top_right_coords, bottom_left_coords)
    cropped_img.save(cropped_output_path)
    print(f"裁剪后的图像已保存到 {cropped_output_path}")
