from PIL import Image
import os

def resize_images(input_folder, output_folder, target_size):
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):  # 只处理图像文件
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            # 打开图像文件
            with Image.open(input_path) as img:
                # 调整图像大小
                resized_img = img.resize(target_size)

                # 保存调整大小后的图像
                resized_img.save(output_path)

# 示例用法
input_folder = '/home/user/software/freeplussparse/data/DTU/Rectified/scan21'
output_folder = '/home/user/software/freeplussparse/data/DTU/Rectified/scan21copy'
target_size = (1600, 1200)  # 目标尺寸，例如 (width, height)

resize_images(input_folder, output_folder, target_size)