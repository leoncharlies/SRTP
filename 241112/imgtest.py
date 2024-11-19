from PIL import Image
import matplotlib.pyplot as plt
Image.MAX_IMAGE_PIXELS = None  # 或者可以设置为特定的值

def view_image_region(image_path, top_left, bottom_right):
    """
    根据给定的像素范围打开图像并显示该区域。

    :param image_path: 图像文件路径
    :param top_left: 左上角坐标 (x1, y1)
    :param bottom_right: 右下角坐标 (x2, y2)
    """
    # 打开图像
    img = Image.open(image_path)

    # 截取指定区域
    region = img.crop((top_left[0], top_left[1], bottom_right[0], bottom_right[1]))

    # 显示截取的区域
    plt.imshow(region)
    plt.axis('off')  # 不显示坐标轴
    plt.title(f"显示区域: {top_left} 到 {bottom_right}")
    plt.show()



