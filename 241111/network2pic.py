import pandas as pd
from PIL import Image, ImageDraw
from tqdm import tqdm
import os

def draw_lines_from_csv(csv_file, output_folder, img_width=8000, img_height=8000):#图像分辨率不能设太高，不然运行速度太慢，像素16000*16000会要四十分钟左右。

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    df = pd.read_csv(csv_file)

    grouped = df.groupby('image_id')

    for image_id, group in tqdm(grouped, desc="处理图像"):

        img = Image.new('RGB', (img_width, img_height), 'white')
        draw = ImageDraw.Draw(img)
        
        # 找到该组数据的经纬度范围，用于缩放
        min_lon = min(group['start_lon'].min(), group['end_lon'].min())
        max_lon = max(group['start_lon'].max(), group['end_lon'].max())
        min_lat = min(group['start_lat'].min(), group['end_lat'].min())
        max_lat = max(group['start_lat'].max(), group['end_lat'].max())

        lon_margin = (max_lon - min_lon) * 0.1
        lat_margin = (max_lat - min_lat) * 0.1
        min_lon -= lon_margin
        max_lon += lon_margin
        min_lat -= lat_margin
        max_lat += lat_margin
        
        def transform_coordinates(lon, lat):

            x = int((lon - min_lon) / (max_lon - min_lon) * img_width)

            y = int((max_lat - lat) / (max_lat - min_lat) * img_height)
            return x, y
        

        for _, row in group.iterrows():
            start_pixel = transform_coordinates(row['start_lon'], row['start_lat'])
            end_pixel = transform_coordinates(row['end_lon'], row['end_lat'])
            
            draw.line([start_pixel, end_pixel], fill='black', width=5)
            
            # 在起点和终点画点
            point_radius = 10
            draw.ellipse([start_pixel[0]-point_radius, start_pixel[1]-point_radius,
                         start_pixel[0]+point_radius, start_pixel[1]+point_radius],
                        fill='red')
            draw.ellipse([end_pixel[0]-point_radius, end_pixel[1]-point_radius,
                         end_pixel[0]+point_radius, end_pixel[1]+point_radius],
                        fill='blue')

        output_path = os.path.join(output_folder, f'image_{image_id}.png')
        img.save(output_path)
        print(f"已保存图像: {output_path}")

if __name__ == '__main__':
    csv_file_path = '../origindata/road_cor.csv'  
    output_folder_path = '../241111/output_images'  
    draw_lines_from_csv(csv_file_path, output_folder_path)