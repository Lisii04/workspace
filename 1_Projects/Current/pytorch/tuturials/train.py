from datasets import ArmorData
from models import LeNet5

import torch
from torchvision.transforms import ToTensor
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchsummary import summary

#常数和算法定义区
transform=transforms.Compose(
    [ToTensor(),
    transforms.Resize((36,36)),
    transforms.RandomRotation((-10,10))
    ])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
classes={'0':0,'1':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'A':10}

#数据集加载
f=open("./data_v3.txt")
root_path="./data_v3/"
train_image_paths=[]
train_labels=[]
test_image_paths=[]
test_labels=[]
line=f.readline()
#读取并且加载图像地址和标签
while(line):
    img_pair=line.split(' ')  #img_pair[0]是图片的地址，img_pair[1][0]是label
    img_address=root_path+img_pair[0]
    if(img_pair[1][0]!='0'):#不是负样本
        if(int(img_pair[0][3:7])>1250):
            test_image_paths.append(img_address)
            test_labels.append((classes[img_pair[1][0]]))
        else:
            train_image_paths.append(img_address)
            train_labels.append((classes[img_pair[1][0]]))
    else:#是负样本
        if(int(img_pair[0][2:7])>10992):
            test_image_paths.append(img_address)
            test_labels.append((classes[img_pair[1][0]]))
        else:
            train_image_paths.append(img_address)
            train_labels.append((classes[img_pair[1][0]]))
    line=f.readline()

#加载数据集
train_dataset=ArmorData(train_image_paths,train_labels,transform=transform)
test_dataset=ArmorData(test_image_paths,test_labels,transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

#检查模型输出探针
for batch_train in tqdm(train_dataloader,desc="Loading training data",leave=False):
    pass
for batch_test in tqdm(test_dataloader,desc="Loading test data",leave=False):
    pass


for images, labels in train_dataloader:
    # access the first image and its corresponding label
    image = images[2]
    label = labels[2]
    image = np.transpose(image.numpy(), (1, 2, 0))
    plt.imshow(image)
    plt.title(f"Label: {label}")
    plt.show()
    break  # break after processing the first element
#数据信息
num_testbatches=len(test_dataloader)
test_size=len(test_dataloader.dataset)
num_trainbatches=len(train_dataloader)
train_size=len(train_dataloader.dataset)
best_path="best.pt"
last_path="last.pt"
#Lenet-5模型构建
model=LeNet5(num_classes=11)
model=model.to(device)
summary(model,(1,36,36))
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()
num_epochs=15
score=0.0

#tensorboard初始化
writer=SummaryWriter('./runs')
print("start with tensorboard --logdir=runs, view at http://localhost:6006/")
#训练
for epoch in range(num_epochs):  # loop over the dataset multiple times
    print(epoch)
    correct = 0
    total = 0
    train_loss=0.0
    test_loss=0.0
    train_accuracy=0.0
    test_accuracy=0.0
    model=model.to(device)# 转移到gpu
    model.train()
    for data in tqdm(train_dataloader,desc=f"Epoch {epoch+1}/{num_epochs}",leave=True):
        # get the inputs; data is a list of [inputs, labels]
        train_img, train_labels = data
        train_img, train_labels = train_img.to(device), train_labels.to(device)
        print(train_img.shape)
        optimizer.zero_grad()
        outputs = model(train_img)
        loss = criterion(outputs, train_labels)
        train_accuracy+=(outputs.argmax(1) == train_labels).type(torch.float).sum().item()
        train_loss+=loss
        loss.backward()
        optimizer.step()
        model.eval()
    with torch.no_grad():
        for img,labels in tqdm(test_dataloader,desc=f"Epoch {epoch+1}/{num_epochs}",leave=True):
            img=img.to(device)
            labels=labels.to(device)
            label_pred=model(img)
            test_loss+=criterion(label_pred,labels)
            test_accuracy += (label_pred.argmax(1) == labels).type(torch.float).sum().item()
    train_loss/=num_trainbatches
    train_accuracy/=train_size
    test_loss /= num_testbatches
    test_accuracy /= test_size
    writer.add_scalar("training loss",train_loss,epoch+1)
    writer.add_scalar("Val loss", test_loss, epoch + 1)
    writer.add_scalar("train accuracy", train_accuracy, epoch + 1)
    writer.add_scalar("test accuracy", test_accuracy, epoch + 1)
    tqdm.write(
            f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f} - Train Acc: {train_accuracy:.4f} - Val Loss: {test_loss:.4f} - Val Acc: {test_accuracy:.4f}")
    if(train_accuracy+test_accuracy)>score:
        score=train_accuracy+test_accuracy
        torch.save(model.state_dict(),best_path)
    elif(epoch==num_epochs-1):
        torch.save(model.state_dict(),last_path)
    else:
        pass
    #writer.flush()
writer.close()
print('Finished Training')