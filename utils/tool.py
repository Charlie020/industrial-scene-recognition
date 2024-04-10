import torch
from d2l import torch as d2l
from IPython import display
from matplotlib import pyplot as plt
from torch import nn
from pathlib import Path
import numpy as np
import cv2
import os


def get_dataloader_workers():
    return 4


# 模型相关信息
def net_info(net, img, verbose=False):
    n_p = sum(x.numel() for x in net.parameters())
    n_g = sum(x.numel() for x in net.parameters() if x.requires_grad)
    n_l = len(list(net.modules()))
    from thop import profile
    from copy import deepcopy
    flops = profile(deepcopy(net), inputs=(img,), verbose=verbose)[0] / 1e9 * 2
    print(f"Model Summary: {n_l} layers, {n_p} parameters, {n_g} gradients, {flops:.2f} GFLOPs")


# Xavier初始化模型参数
def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)


# 加载网络参数
def load_paramsToNet(net, params_file):
    if params_file == '' or Path(params_file).exists() is False:
        print("initializing network parameters")
        net.apply(init_weights)
    else:
        print("loading parameters to the network")
        net.load_state_dict(torch.load(params_file))


# 获得保存结果的路径
def get_save_path():
    folder_name = "runs"
    exp_name = "exp"
    base_path = Path(folder_name) / exp_name

    path = base_path
    index = 1
    while path.exists():
        path = base_path.parent / (base_path.name + str(index))
        index += 1

    return path


def get_rgb_mean_std(dataset_path, train_txt, resize=(32, 32)):
    with open(train_txt, 'r') as f:
        # 读每一行，去掉|0或|1
        img_path_list = [line[:-3] for line in f.readlines()]
        len_ = len(img_path_list)

    img_list = []
    for i, img_path in enumerate(img_path_list):
        img = cv2.imread(os.path.join(dataset_path + "/", img_path))
        img = cv2.resize(img, resize)
        img = img[:, :, :, np.newaxis]
        img_list.append(img)
        print(i + 1, '/', len_)

    print("calculating...")
    imgs = np.concatenate(img_list, axis=3)
    imgs = imgs.astype(np.float32) / 255.
    means, stds = [], []
    for i in range(3):
        pixels = imgs[:, :, i, :].ravel()
        means.append(np.mean(pixels))
        stds.append(np.std(pixels))

    means.reverse()
    stds.reverse()
    print("normMean = {}".format(means))
    print("normStd = {}".format(stds))


def get_fps(net, test_iter, device):
    if isinstance(net, torch.nn.Module):
        net.eval()

    sum_test_img = 0
    for _, y in test_iter:
        sum_test_img += len(y)

    timer = d2l.Timer()
    timer.start()
    with torch.no_grad():
        for X, _ in test_iter:
            X = X.to(device)
            net(X)
    timer.stop()

    return int(sum_test_img / timer.sum())


def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = d2l.argmax(y_hat, axis=1)
    cmp = d2l.astype(y_hat, y.dtype) == y
    return float(d2l.reduce_sum(d2l.astype(cmp, y.dtype)))


def evaluate_accuracy(net, data_iter, device):
    if isinstance(net, torch.nn.Module):
        net.eval()
    net.to(device)

    metric = Accumulator(2)

    with torch.no_grad():
        for X, y in data_iter:
            X, y = X.to(device), y.to(device)
            metric.add(accuracy(net(X), y), d2l.size(y))
    return metric[0] / metric[1]


def train_ch6(net, train_iter, test_iter, num_epochs, device):
    net.to(device)
    print('training on', device)

    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    loss = nn.CrossEntropyLoss()
    animator = Animator(xlabel='epoch', xlim=[0, num_epochs],
                        legend=['train loss', 'train acc', 'test acc'])
    timer, num_batches = d2l.Timer(), len(train_iter)
    max_test_acc, max_test_acc_epoch = -1.0, 0
    save_dir = get_save_path()
    model_save_path = save_dir / 'weights'
    model_save_path.mkdir(parents=True, exist_ok=True)
    log_path = save_dir / 'log.txt'

    for epoch in range(num_epochs):
        metric = Accumulator(3)
        net.train()
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                # 分类错误 / 分类正确 / Batchsize
                metric.add(l * X.shape[0], accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (train_l, train_acc, None))
        test_acc = evaluate_accuracy(net, test_iter, device)
        animator.add(epoch + 1, (None, None, test_acc))
        animator.save_fig(save_dir / 'result.png')
        # 保存最好的模型
        if max_test_acc < test_acc:
            max_test_acc = test_acc
            max_test_acc_epoch = epoch + 1
            torch.save(net.state_dict(), model_save_path / 'best.pt')
        log_msg = f'epoch = {epoch + 1}        train loss = {train_l:.3f}        train acc = {train_acc:.3f}        test acc = {test_acc:.3f}\n'
        print(log_msg)
        with open(log_path, 'a') as file:
            file.write(log_msg)

    # 保存最后的模型
    torch.save(net.state_dict(), model_save_path / 'last.pt')
    log_msg = (
        f'max test acc = {max_test_acc:.3f} in epoch = {max_test_acc_epoch}\n'
        f'{int(metric[2] * num_epochs)} examples for {timer.sum() / 60:.1f} minutes on {str(device)}\n'
        f'FPS: {get_fps(net, test_iter, device)}\n'
    )
    print(log_msg)
    with open(log_path, 'a') as file:
        file.write(log_msg)


class Animator:
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(4, 3)):
        if legend is None:
            legend = []
        d2l.use_svg_display()
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # Use a lambda function to capture arguments
        self.config_axes = lambda: d2l.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        # 动态绘图
        # --- start ---
        plt.draw()
        plt.pause(0.1)
        # --- end ---
        display.clear_output(wait=True)

    def save_fig(self, path):
        plt.savefig(path)


class Accumulator:
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
