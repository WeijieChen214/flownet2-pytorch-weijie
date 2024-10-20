import torch
import numpy as np
import argparse
import cv2
import torch, gc
from models import FlowNet2  # the path is depended on where you create this module
from utils.frame_utils import read_gen  # the path is depended on where you create this module
import os


# 确保两张图片的尺寸一致
def resize_images_to_same_size(img1, img2):
    height1, width1 = img1.shape[:2]
    height2, width2 = img2.shape[:2]

    # 如果图片尺寸不同，则将第二张图片调整为与第一张图片相同的尺寸
    if height1 != height2 or width1 != width2:
        img2 = cv2.resize(img2, (width1, height1), interpolation=cv2.INTER_AREA)

    return img1, img2


def resize_to_target(img):
    # 目标尺寸 (256, 480)
    target_height, target_width = 256, 480

    # 调整图片大小
    resized_img = cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_AREA)

    return resized_img



# def resize_to_multiple_of_16(path, output_path):
#     # 读取图片
#     img = cv2.imread(path)
#     # 获取图片的尺寸
#     height, width = img.shape[:2]
#
#     # 计算新的尺寸，使其成为16的倍数
#     new_width = (width // 16) * 16
#     new_height = (height // 16) * 16
#
#     # 裁剪图片
#     img = img[:new_height, :new_width]
#
#     # 或者可以选择填充图片
#     # resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
#     # 填充图片以使其尺寸成为16的倍数
#     # top = (new_height - height) // 2
#     # bottom = new_height - height - top
#     # left = (new_width - width) // 2
#     # right = new_width - width - left
#     # color = [0, 0, 0]  # 填充颜色为黑色
#     # img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
#
#     # 保存调整后的图片
#     cv2.imwrite(output_path, img)


if __name__ == '__main__':
    # obtain the necessary args for construct the flownet framework
    gc.collect()
    torch.cuda.empty_cache()
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    parser = argparse.ArgumentParser()
    parser.add_argument('--fp16', action='store_true', help='Run model in pseudo-fp16 mode (fp16 storage fp32 math).')
    parser.add_argument("--rgb_max", type=float, default=255.)
    
    args = parser.parse_args()

    # initial a Net
    with torch.no_grad():
        net = FlowNet2(args).cuda()

    #print(net)
    #print(net.out_conv3_1.shape, net.out_deconv3.shape, net.flow4_up.shape)

    # load the state_dict
    dict = torch.load("FlowNet2_checkpoint.pth.tar")
    net.load_state_dict(dict["state_dict"])

    # load the image pair, you can find this operation in dataset.py
    pim1 = read_gen("datasets/MPI-Sintel/training/flow/output_0031.jpg")
    pim2 = read_gen("datasets/MPI-Sintel/training/flow/output_0032.jpg")

    # 调整图片尺寸一致
    #pim1, pim2 = resize_images_to_same_size(pim1, pim2)

    # 调整图片尺寸为16的倍数
    pim1 = resize_to_target(pim1)
    pim2 = resize_to_target(pim2)

    print(pim1.shape)
    print(pim2.shape)

    #resize_to_multiple_of_16("datasets/MPI-Sintel/training/flow/output_0001.png","datasets/MPI-Sintel/training/flow/output_0002.png")
    images = [pim1, pim2]
    images = np.array(images).transpose(3, 0, 1, 2)
    im = torch.from_numpy(images.astype(np.float32)).unsqueeze(0).cuda()

    # process the image pair to obtian the flow
    result = net(im).squeeze()


    # save flow, I reference the code in scripts/run-flownet.py in flownet2-caffe project
    def writeFlow(name, flow):
        f = open(name, 'wb')
        f.write('PIEH'.encode('utf-8'))
        np.array([flow.shape[1], flow.shape[0]], dtype=np.int32).tofile(f)
        flow = flow.astype(np.float32)
        flow.tofile(f)
        f.flush()
        f.close()


    data = result.data.cpu().numpy().transpose(1, 2, 0)
    writeFlow("output/image.flo", data)
