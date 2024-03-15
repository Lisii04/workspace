import torch
import torch.nn as nn
import torchvision.models as models

# 加载预训练模型，例如 ResNet
pretrained_model = models.resnet18(pretrained=True)

# 解冻所有参数
for param in pretrained_model.parameters():
    param.requires_grad = True

# 替换顶部分类层（适用于 ResNet）
num_classes = 10
pretrained_model.fc = nn.Linear(pretrained_model.fc.in_features, num_classes)

# 定义优化器，优化所有参数
optimizer = torch.optim.SGD(pretrained_model.parameters(), lr=0.001, momentum=0.9)

# 进行微调训练
# 在每个训练迭代中，计算损失，执行反向传播，更新权重
# 注意：确保你有相应的训练数据和损失函数来完成微调训练
