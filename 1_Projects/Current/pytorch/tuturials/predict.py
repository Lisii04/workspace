import cv2
import numpy as np
import torch
from models import LeNet5

classes={0:'0',1:'1',2:'2',3:'3',4:'4',5:'5',6:'6',7:'7',8:'8',9:'9','10':'A'}

kernel=np.ones((1,1),np.uint8)
img=cv2.imread("T4.png")
img=cv2.resize(img,(36,36))
img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
_,img=cv2.threshold(img,127,255,cv2.THRESH_BINARY)
img=cv2.erode(img,kernel)
img=cv2.GaussianBlur(img,(3,3),0)
img=torch.from_numpy(img).unsqueeze(0).unsqueeze(0)
img=img.to(torch.float32)

model_dict=torch.load("best.pt",map_location=torch.device('cpu'))
model=LeNet5()
model.load_state_dict(model_dict)
model.eval()
with torch.no_grad():
    pred=model(img)
print(pred)
