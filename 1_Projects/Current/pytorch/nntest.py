import cv2
import matplotlib
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.utils.data.dataset as Dataset
import torch.utils.data.dataloader as DataLoader
import torchvision
from torchvision.transforms import ToTensor
from torchvision import transforms
# 模型定义
class Simple_model(nn.Module):
    def __init__(self,input_channels=1,output_channels=1):
        super(Simple_model,self).__init__()
        self.layer1_conv = nn.Conv2d(input_channels,output_channels,kernel_size=3,stride=1,padding=1,bias=True)
        self.fc1 = nn.Linear(1*28*28,100)
        self.fc2 = nn.Linear(100,10)
    def forward(self,x):
        out = F.relu(self.layer1_conv(x))
        out = out.view(-1,1*28*28)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out
    
class Simple_dataset(Dataset.Dataset):
    def __init__(self,image_paths,label,transform):
        self.image_paths=image_paths
        self.label=label
        self.transform=transform
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, index, shape=(28,28)):
        img = cv2.imread(self.image_paths[index])
        img = cv2.resize(img,shape)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        _,img = cv2.threshold(img,127,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)
        if self.transform:
            img = self.transform(img)
        else:
            img = torch.from_numpy(img)
        label = torch.tensor(self.label[index])
        return img, label

# 训练

#常数和算法定义区
transform=transforms.Compose(
    [ToTensor(),
    transforms.Resize((36,36)),
    transforms.RandomRotation((-10,10))
    ])


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
classes={'0':0,'1':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9}

net = Simple_model()
print(net.to(device))