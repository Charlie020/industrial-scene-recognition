import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


# 读取数据集，并分为 len / batch_size 个批次
def dataloader(dataset_path, file_path, batch_size, num_workers, device, resize=None, train=True):
    trans = [transforms.ToTensor(), transforms.Normalize(mean=[0.452, 0.436, 0.408], std=[0.266, 0.263, 0.279])]
    if train:
        trans.insert(0, transforms.RandomResizedCrop(224))
        trans.insert(0, transforms.RandomHorizontalFlip(0.5))
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)

    dataset = MyDataset(dataset_path=dataset_path, txt_path=file_path, transform=trans)

    if device == torch.device("cuda"):
        dataset_iter = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        print("dataset is loaded into cuda")
    else:
        dataset_iter = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        print("dataset is loaded into cpu")

    return dataset_iter


class MyDataset(Dataset):
    def __init__(self, dataset_path, txt_path, transform=None):
        self.dataset_path = dataset_path
        self.txt_path = txt_path
        # 获取图片及标签
        self.data = self.read_image_label()
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, label = self.data[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(int(label))
        return image, label

    # 按txt文件中每行读取，并与数据集路径进行拼接
    def read_image_label(self):
        with open(self.txt_path, 'r') as file:
            data = [line.strip().split('|') for line in file.readlines()]

        # 下面的代码得到的data中的每一项是一个list，其中list[0]是图片的绝对路径，list[1]是0/1
        data = [[self.dataset_path + '/' + item[0]] + item[1:] for item in data]
        return data
