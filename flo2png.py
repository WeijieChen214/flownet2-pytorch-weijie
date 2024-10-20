import os
import numpy as np
from PIL import Image

def load_flow_to_numpy(input):
    # magic = np.fromfile(input, np.float32, count=1)
    # assert (202021.25 == magic), 'Magic number incorrect. Invalid .flo file'
    # h = np.fromfile(input, np.int32, count=1)[0]  # 图像高
    # w = np.fromfile(input, np.int32, count=1)[0]  # 图像宽
    # data = np.fromfile(input, np.float32, count=2 * w * h)  # 每个像素点有 2 个浮点数，分别表示该点在 x 和 y 方向上的运动（即光流向量）
    # data2D = np.resize(data, (w, h, 2))  # 宽，高，两通道（x 和 y 方向）
    data2D = input  # 假设 input 已经是形状为 (height, width, 2) 的数组
    return data2D

def load_flow_to_png(input):
    flow = load_flow_to_numpy(input)  # 加载光流数据并转换为 NumPy 数组
    image1, image2 = flow_to_image_separate(flow)
    return image1, image2

# 光流（flow）数据转换为灰度图像
def flow_to_image_separate(flow, max_flow=256):
    if max_flow is not None:
        max_flow = max(max_flow, 1.)
    else:
        max_flow = np.max(flow)

    u, v = flow[:, :, 0], flow[:, :, 1]

    # x方向分量（水平分量 u）
    mag_u = np.abs(u)  # 水平分量的大小
    img_u = (mag_u / np.max(mag_u) * 255).astype(np.uint8)  # 映射到0-255的灰度图

    # y方向分量（垂直分量 v）
    mag_v = np.abs(v)  # 垂直分量的大小
    img_v = (mag_v / np.max(mag_v) * 255).astype(np.uint8)  # 映射到0-255的灰度图

    return img_u, img_v  # 返回x方向和y方向的两张图片


def flow2image(input, output_folder,index):
    # 构建完整路径

    # 读取光流数据并转换为灰度图像
    image1, image2 = load_flow_to_png(input)

    img_u_pil = Image.fromarray(image1)
    img_v_pil = Image.fromarray(image2)


    img_u_pil.save(os.path.join(output_folder, f"{index}_x.jpg"))
    img_v_pil.save(os.path.join(output_folder, f"{index}_y.jpg"))



