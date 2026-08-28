import torch.nn as nn
import torchvision.models as models
import torch

def get_resnet34():
    try:
        model = models.resnet34(weights=None)
    except TypeError:
        model = models.resnet34(pretrained=False)
    model.avgpool = nn.Identity()
    model.fc = nn.Identity()
    return model

class Res34_encoder(nn.Module):
    # 定义初始化函数
    def __init__(self, out_channel = 2):
        # 调用 nn.Module 的初始化函数
        super(Res34_encoder, self).__init__()

        # 创建 ResNet34 模型
        self.resnet = get_resnet34()
        # 定义 layer0，包括 ResNet34 的第一层卷积、批归一化、ReLU 和最大池化层
        self.layer0 = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu,
            self.resnet.maxpool
        )

        # 定义 Encode 部分，包括 ResNet34 的 layer1、layer2、layer3 和 layer4
        self.layer1 = self.resnet.layer1
        self.layer2 = self.resnet.layer2
        self.layer3 = self.resnet.layer3
        self.layer4 = self.resnet.layer4

        # 定义 Bottleneck 部分，包括两个卷积层、ReLU、批归一化和最大池化层
        self.bottleneck = nn.Sequential(
            nn.Conv2d(kernel_size=(3, 3), in_channels=512, out_channels=1024, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(1024),
            nn.Conv2d(kernel_size=(3, 3), in_channels=1024, out_channels=1024, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(1024),
            # nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.droupout = nn.Dropout(0.5)
        self.linear  = nn.Linear(1024,out_channel)

    def forward(self,x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.bottleneck(x)
        print(x.shape)
        x = self.avgpool(x)
        x = torch.flatten(x,start_dim=1)
        x = self.droupout(x)
        x = self.linear(x)
        return x

class Res34_pruningencoder(nn.Module):
    # 定义初始化函数
    def __init__(self, out_channel = 2):
        # 调用 nn.Module 的初始化函数
        super(Res34_pruningencoder, self).__init__()

        # 创建 ResNet34 模型
        self.resnet = get_resnet34()
        # 定义 layer0，包括 ResNet34 的第一层卷积、批归一化、ReLU 和最大池化层
        self.layer0 = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu,
            self.resnet.maxpool
        )

        # 定义 Encode 部分，包括 ResNet34 的 layer1、layer2、layer3 和 layer4
        self.layer1 = self.resnet.layer1
        self.layer2 = self.resnet.layer2
        self.layer3 = self.resnet.layer3
        self.layer4 = self.resnet.layer4

        # # 定义 Bottleneck 部分，包括两个卷积层、ReLU、批归一化和最大池化层
        # self.bottleneck = nn.Sequential(
        #     nn.Conv2d(kernel_size=(3, 3), in_channels=512, out_channels=1024, padding=1),
        #     nn.ReLU(),
        #     nn.BatchNorm2d(1024),
        #     nn.Conv2d(kernel_size=(3, 3), in_channels=1024, out_channels=1024, padding=1),
        #     nn.ReLU(),
        #     nn.BatchNorm2d(1024),
        #     # nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        # )
        self.relu = nn.ReLU()

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.droupout = nn.Dropout(0.6)
        # self.linear1 = nn.Linear(512,128)
        # self.bn1 = nn.BatchNorm1d(128)
        # self.linear2 = nn.Linear(512,256)
        # self.bn2 = nn.BatchNorm1d(256)
        # self.linear3 = nn.Linear(256,128)
        # self.bn3 = nn.BatchNorm1d(128)
        # self.linear4 = nn.Linear(128,64)
        # self.bn4 = nn.BatchNorm1d(64)
        self.linear5 = nn.Linear(512,out_channel)

    def forward(self,x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # x = self.bottleneck(x)
        x = self.avgpool(x)
        x = torch.flatten(x,start_dim=1)

        # x = self.linear1(x)
        x = self.droupout(x)
        # x = self.bn1(x)
        # x = self.relu(x)

        # x = self.linear2(x)
        # x = self.droupout(x)
        # x = self.bn2(x)
        # x = self.relu(x)

        # x = self.linear3(x)
        # x = self.droupout(x)
        # x = self.bn3(x)
        # x = self.relu(x)

        # x = self.linear4(x)
        # x = self.droupout(x)
        # x = self.bn4(x)
        # x = self.relu(x)

        x = self.linear5(x)
        return x

if __name__ == '__main__':
    # model = Res34_encoder()
    model = Res34_pruningencoder()
    input = torch.rand(8,3,256,256)
    output = model(input)
    print(output.shape)