import os
import torch
import numpy as np

from torch import nn
from PIL import Image
from d2l.torch import d2l
from torchvision import transforms

from utils.model import ResNet18
from utils.tool import net_info, load_paramsToNet


def detect_img(net, img_path, trans, device):
    assert img_path.endswith('jpg') or img_path.endswith('png')

    img = trans(Image.open(img_path).convert('RGB'))
    img = img[np.newaxis, :]

    with torch.no_grad():
        img = img.to(device)
        y_hat = d2l.argmax(net(img), axis=1).item()
        print(f'{img_path}: {y_hat}')


def detect(net, img_path='', folder_path='', resize=None, device=torch.device("cpu")):
    if isinstance(net, nn.Module):
        net.eval()
    net.to(device)

    trans = [transforms.ToTensor(), transforms.Normalize(mean=[0.452, 0.436, 0.408], std=[0.266, 0.263, 0.279])]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)

    if img_path != '':
        detect_img(net, img_path, trans, device)
    if folder_path != '':
        for filename in os.listdir(folder_path):
            img_path = os.path.join(folder_path, filename)
            detect_img(net, img_path, trans, device)


if __name__ == '__main__':
    # 模型
    net = ResNet18()

    # 打印相关信息
    net_info(net, torch.randn(1, 3, 224, 224))

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 读取模型参数
    load_paramsToNet(net, r'runs/ResNet18_300/weights/best.pt')

    # 检测
    img_path = ''
    folder_path = r'Data/places365_standard/train/basement'
    detect(net, img_path, folder_path, 224, device)

