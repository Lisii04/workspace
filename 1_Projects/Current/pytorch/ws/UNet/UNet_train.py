import glob
import os
import argparse
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms    # torchvision包含计算机视觉的常用工具
from torch.autograd import Variable
from tqdm import tqdm
from UNet import Unet
import PIL.Image as Image
from torch.utils.data import Dataset

t = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])


class Unet_dataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __getitem__(self, index):
        img = cv2.imread(self.image_paths[index], cv2.IMREAD_GRAYSCALE).astype(
            np.float32) / 255.0
        img = cv2.resize(img, (224, 224))
        mask_img = cv2.imread(self.mask_paths[index], cv2.IMREAD_GRAYSCALE).astype(
            np.float32) / 255.0
        mask_img = cv2.resize(mask_img, (224, 224))

        return img, mask_img

    def __len__(self):
        return len(self.image_paths)


def load_data(train_batch_size, test_batch_size):
    # 获取MNIST数据集

    kwargs = {}

    training_image_paths = glob.glob(
        '/workspaces/workspace/1_Projects/Current/pytorch/ws/data/training/images/*.tif')
    training_mask_paths = glob.glob(
        '/workspaces/workspace/1_Projects/Current/pytorch/ws/data/training/1st_manual/*.png')

    test_image_paths = glob.glob(
        '/workspaces/workspace/1_Projects/Current/pytorch/ws/data/test/images/*.tif')
    test_mask_paths = glob.glob(
        '/workspaces/workspace/1_Projects/Current/pytorch/ws/data/test/1st_manual/*.png')

    # 获取训练数据
    train_loader = torch.utils.data.DataLoader(
        Unet_dataset(training_image_paths,
                     training_mask_paths, transform=t),
        batch_size=train_batch_size, shuffle=True, **kwargs)

    # 获取测试数据
    test_loader = torch.utils.data.DataLoader(
        Unet_dataset(test_image_paths,
                     test_mask_paths, transform=t),
        batch_size=test_batch_size, shuffle=True, **kwargs)

    return (train_loader, test_loader)


def train(model, optimizer, epoch, train_loader, log_interval):
    # 声明正在训练模型
    model.train()

    # 遍历数据批次
    for batch_idx, (data, target) in tqdm(enumerate(train_loader)):
        # 将输入和目标输出包装在`Variable`中
        data, target = Variable(data), Variable(target)

        data = torch.unsqueeze(data, 1)
        target = target.long()

        print(data.shape)
        print(target.shape)
        print(target.min(), target.max())

        # 清除梯度，因为PyTorch会累积梯度
        optimizer.zero_grad()

        # 前向传播
        output = model(data)

        # 计算负对数似然损失
        # loss = F.nll_loss(output, target)
        loss = nn.BCEWithLogitsLoss(output, target)

        # 反向传播
        loss.backward()

        # 更新梯度
        optimizer.step()

        # 输出调试信息
        if batch_idx % log_interval == 0:
            print('训练 Epoch: {} [{}/{} ({:.0f}%)]\t损失: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data.item()))


def test(model, test_loader):
    # 声明正在测试模型；这会阻止例如Dropout等层起作用
    model.eval()

    # 初始化损失和正确预测的累加器
    test_loss = 0
    correct = 0

    # 使用`torch.no_grad()`优化验证过程
    with torch.no_grad():
        # 遍历数据
        for data, target in test_loader:  # 在`torch.no_grad()`下，不需要将数据和目标包装在`Variable`中
            # 获取输出
            output = model(data)

            # 计算并累加损失
            test_loss += F.nll_loss(output, target,
                                    reduction='sum').data.item()

            # 获取最大对数概率的索引(预测的输出标签）
            pred = output.data.argmax(1)

            # 如果预测正确，则增加正确预测的累加器
            correct += pred.eq(target.data).sum()

    # 打印平均测试损失
    test_loss /= len(test_loader.dataset)
    print('\n测试集: 平均损失: {:.4f}, 准确率: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


if __name__ == '__main__':
    # 从命令行选项设置训练设置，或使用默认值
    parser = argparse.ArgumentParser(description='PyTorch MNIST 示例')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='训练时的输入批次大小(默认值: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='测试时的输入批次大小(默认值: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='训练的轮数(默认值: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='学习率(默认值: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD的动量(默认值: 0.5)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='随机种子(默认值: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='日志输出间隔批次数')
    args = parser.parse_args()

    # 为伪随机数生成器提供种子，以便可以重现相同的结果
    torch.manual_seed(args.seed)

    # 实例化模型
    model = Unet()

    # 选择SGD作为优化器，并使用参数和设置进行初始化
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum)

    # 加载数据
    train_loader, test_loader = load_data(
        args.batch_size, args.test_batch_size)

    # 训练和测试模型
    for epoch in range(1, args.epochs + 1):
        train(model, optimizer, epoch, train_loader,
              log_interval=args.log_interval)
        test(model, test_loader)

    # 保存模型以备将来使用
    package_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(package_dir, 'model')
    torch.save(model.state_dict(), model_path+".pt")
