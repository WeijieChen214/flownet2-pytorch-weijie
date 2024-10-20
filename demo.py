import os
from PIL import Image

# 指定读取图片的文件夹路径和保存缩放后的图片路径
input_dir = "C:/Users/CWJ/Desktop/dataset/demo"
output_dir = "C:/Users/CWJ/Desktop/dataset/demo/resize"

# 确保输出文件夹存在
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 目标尺寸
target_size = (320, 180)

# 获取图片列表
image_files = [f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

# 循环处理每张图片
for image_file in image_files:
    # 打开图片
    image_path = os.path.join(input_dir, image_file)
    img = Image.open(image_path)

    # 获取图片的原始尺寸
    width, height = img.size

    # 计算缩放比例，保持宽高比不变
    scale = min(target_size[0] / width, target_size[1] / height)
    new_size = (int(width * scale), int(height * scale))

    # 按比例缩放图片
    resized_img = img.resize(new_size, Image.Resampling.LANCZOS)

    # 创建一个新的 320x180 的背景，并将缩放后的图片粘贴到中心
    new_img = Image.new("RGB", target_size, (0, 0, 0))  # 黑色背景，可以改成其他颜色
    paste_position = ((target_size[0] - new_size[0]) // 2, (target_size[1] - new_size[1]) // 2)
    new_img.paste(resized_img, paste_position)

    # 保存缩放后的图片
    save_path = os.path.join(output_dir, image_file)
    new_img.save(save_path)

    print(f"Image {image_file} resized and saved to {save_path}.")
