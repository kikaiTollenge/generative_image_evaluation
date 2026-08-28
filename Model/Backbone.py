import torch
import torchvision.models as models
import torch.nn as nn
from Model.DeepLabV3.resnet_atrous import Atrous_resnet50_os16
from Model.DeepLabV3.aspp import ASPP_Bottleneck
class resnet50(nn.Module):
    def __init__(self):
        super().__init__()
        model = models.resnet50(weights = None)
        self.backbone = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
        )
    def forward(self,x):
        return (self.backbone(x),)

class deeplabv3plusencoder(nn.Module):  # 这个效果很差不能用
    def __init__(self):
        super().__init__()
        self.layer1 = Atrous_resnet50_os16()
        self.layer2 = ASPP_Bottleneck()

    def forward(self,x):
        x = self.layer1(x)[-1]
        x = self.layer2(x)
        return (x,)


if __name__ == '__main__':
    input = torch.randn((2,3,256,256))
    model = deeplabv3plusencoder()
    output = model(input)[-1]
    print(output.shape)