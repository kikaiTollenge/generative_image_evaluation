import torch.nn as nn
import torchvision.models as models
import torch

# pytorch官方的空洞resnet50
def modify_resnet50_atrous(resnet50):
    for _, layer in enumerate([resnet50.layer3, resnet50.layer4]):
        if layer == resnet50.layer3:
            layer[0].conv2 = nn.Conv2d(256,256,3,1,1)
            layer[0].downsample[0] = nn.Conv2d(512, 1024, 1, 1)
            for i in range(1, len(layer)):
                layer[i].conv2 = nn.Conv2d(256, 256, 3, 1, 2, 2)
        if layer == resnet50.layer4:
            for i in range(1, len(layer)):
                layer[i].conv2 = nn.Conv2d(512, 512, 3, 1, 4, 4)
    resnet50.avgpool = nn.Identity()
    resnet50.fc = nn.Identity()
    return resnet50


class Atrous_resnet50_os16(nn.Module):
    def __init__(self):
        super().__init__()
        model = modify_resnet50_atrous(models.resnet50())
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        # layer4中没有使用2*(1,2,4)的multi-grid
    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        low_feature = x
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return (low_feature,x)



if __name__ == '__main__':
    # print(models.resnet50(weights = None))
    resnet = Atrous_resnet50_os16()
    print(resnet)
    x = torch.randn(1, 3, 256, 256)
    low_feature,output = resnet(x)
    print(output.shape)
    print(low_feature.shape)
    x = torch.randn(1, 3, 256, 256)
    low_feature,output = resnet(x)
    print(output.shape)
    print(low_feature.shape)