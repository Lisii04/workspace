import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

#参数区
kernel=np.ones((1,1),np.uint8)

#建立类和对象区
class ArmorData(Dataset):
    def __init__(self,image_paths,label,transform):#image为图像，label为标签，0为空背景，1为英雄，2为工程，3-5为步兵，6为哨兵
        self.image_paths=image_paths
        self.label=label
        self.transform=transform
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self,index,shape=(36,36)):#此处写预处理函数
        img= cv2.imread(self.image_paths[index])
        img=cv2.resize(img,shape)
        img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        _,img=cv2.threshold(img,127,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)
        img=cv2.erode(img,kernel)
        img=cv2.GaussianBlur(img,(3,3),0)
        #cv2.normalize(img,img,norm_type=cv2.NORM_MINMAX)
        if self.transform:
            img = self.transform(img)
        else:
            img = torch.from_numpy(img)
        label = torch.tensor(self.label[index])
        return img, label

