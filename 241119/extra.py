import csv
import os
import ast
import pandas as pd
from PIL import Image


def parse_pix_list(pix_list_str):
    """
    将 pix_list 字符串解析为字典，返回图片代号和像素点列表。
    """
    pix_dict = ast.literal_eval(pix_list_str)  # 将字符串解析为字典
    if not isinstance(pix_dict, dict):
        raise ValueError("pix_list 数据格式错误")
    image_code = list(pix_dict.keys())[0]  # 获取图片代号
    pixels = pix_dict[image_code]  # 获取像素点列表
    return image_code, pixels


def find_image(image_code, image_folder):
    """
    根据图片代号在指定文件夹中查找图片，返回图片路径。
    """
    for file_name in os.listdir(image_folder):
        if image_code in file_name:
            return os.path.join(image_folder, file_name)
    return None


def calculate_color_ratios(rgb_values, color_ranges):
    """
    计算各颜色像素点占总有效像素点的比例，同时考虑可能出现不属于预定义颜色范围的像素点。
    """
    total_pixels = 0
    color_counts = {color: 0 for color in color_ranges.keys()}
    color_counts["其他"] = 0  # 增加 "其他" 颜色类别计数

    for rgb in rgb_values.values():
        if rgb is None:
            continue  # 跳过无效像素点

        total_pixels += 1  # 统计总有效像素点数

        matched_color = False
        for color, ranges in color_ranges.items():
            r, g, b = rgb
            if (ranges["R"][0] <= r <= ranges["R"][1] and
                ranges["G"][0] <= g <= ranges["G"][1] and
                ranges["B"][0] <= b <= ranges["B"][1]):
                color_counts[color] += 1
                matched_color = True
                break  # 一个像素点只能属于一种颜色

        if not matched_color:
            color_counts["其他"] += 1  # 如果没有匹配的颜色，则归类为 "其他"

    color_ratios = {color: count / total_pixels if total_pixels > 0 else 0
                    for color, count in color_counts.items()}

    return color_ratios


def main(csv_file_path, image_folder, output_csv_path):
    """
    主函数：读取 CSV 文件，计算每条道路对应颜色占比并保存结果。
    """
    # 使用 pandas 读取 CSV 文件
    df = pd.read_csv(csv_file_path)
    
    # 定义颜色范围
    color_ranges = {
        "绿色": {"R": (90, 105), "G": (165, 180), "B": (30, 45)},
        "黄色": {"R": (235, 255), "G": (195, 215), "B": (45, 60)},
        "红色": {"R": (190, 210), "G": (45, 60), "B": (15, 30)},
        "深红色": {"R": (120, 135), "G": (30, 45), "B": (10, 25)}
    }

    image_cache = {}  # 用于缓存已加载的图片

    # 遍历每行数据进行处理
    for row_index, row in df.iterrows():
        try:
            # 解析 pix_list 并查找图片
            pix_list_str = row["pix_list"]
            image_code, pixels = parse_pix_list(pix_list_str)
            image_path = find_image(image_code, image_folder)

            if not image_path:
                print(f"第 {row_index + 1} 行：未找到代号为 {image_code} 的图片")
                continue

            # 从缓存或磁盘加载图片
            if image_path not in image_cache:
                with Image.open(image_path) as img:
                    image_cache[image_path] = img.convert("RGB")
            img = image_cache[image_path]

            rgb_values = {}
            for point in pixels:
                x, y = point
                if 0 <= x < img.width and 0 <= y < img.height:
                    rgb_values[(x, y)] = img.getpixel((x, y))
                else:
                    rgb_values[(x, y)] = None

            # 计算颜色占比
            color_ratios = calculate_color_ratios(rgb_values, color_ranges)

            # 将颜色占比存入 DataFrame
            df.at[row_index, "绿色占比"] = color_ratios.get("绿色", 0)
            df.at[row_index, "黄色占比"] = color_ratios.get("黄色", 0)
            df.at[row_index, "红色占比"] = color_ratios.get("红色", 0)
            df.at[row_index, "深红色占比"] = color_ratios.get("深红色", 0)
        
        except Exception as e:
            print(f"第 {row_index + 1} 行处理时出错: {e}")

    # 将处理结果保存到新的 CSV 文件
    df.to_csv(output_csv_path, index=False, encoding="utf-8")
    print(f"处理完成，结果已保存到 {output_csv_path}")

if __name__ == "__main__":
    
    csv_file = "datas/origindata/output.csv"
    image_folder = "datas/origindata/2024-09-26 00-27"
    output_csv = "datas/origindata/processed_output.csv"
    main(csv_file, image_folder, output_csv)
