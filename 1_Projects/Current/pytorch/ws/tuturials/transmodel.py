from models import LeNet5
import torch.nn as nn

class MyNet(LeNet5):
    def __init__(self,num_classes=10):
        super(MyNet,self).__init__(self,num_classes)
        self.conv4=self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)
    def forward(self,x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu3(x)

if __name__ == '__main__':
    mymodel=MyNet()
    pretrained_model=LeNet5()
    pretrained_model.load_state_dict()