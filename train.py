import torch

from utils.datasets import dataloader
from utils.model import ResNet18
from utils.tool import net_info, load_paramsToNet, get_dataloader_workers, train_ch6, evaluate_accuracy

# 路径
dataset_path = r"Data"
trainset_path = r"Data/train.txt"
testset_path = r"Data/test.txt"

# 超参
batch_size = 16
num_epochs = 300


if __name__ == '__main__':
    net = ResNet18()

    net_info(net, torch.randn(1, 3, 224, 224))  # img_size=(224, 224)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_iter = dataloader(dataset_path, trainset_path, batch_size, get_dataloader_workers(), device, 224)
    test_iter = dataloader(dataset_path, testset_path, batch_size, get_dataloader_workers(), device, 224, train=False)

    load_paramsToNet(net, r'')

    train_ch6(net, train_iter, test_iter, num_epochs, device)

    # test
    print(f'test acc: {evaluate_accuracy(net, test_iter, device):.3f}')



