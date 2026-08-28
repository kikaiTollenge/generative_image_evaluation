from torch import nn
import torchvision.models as models
import torch.nn.functional as F
import torch

class nor_expansive_block(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super(nor_expansive_block, self).__init__()

        # 卷积块的结构
        self.block = nn.Sequential(
            nn.Conv2d(kernel_size=(3, 3), in_channels=in_channels, out_channels=mid_channels, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            nn.Conv2d(kernel_size=(3, 3), in_channels=mid_channels, out_channels=out_channels, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        self.up = nn.ConvTranspose2d(in_channels,in_channels,2,2)

    def forward(self, d, e=None):
        # 拼接
        if e is not None:
            cat = torch.cat([e, d], dim=1)
            out = self.block(cat)
        else:
            out = self.block(d)
        return out

# 定义解码器中的卷积块
class expansive_block(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super(expansive_block, self).__init__()

        # 卷积块的结构
        self.block = nn.Sequential(
            nn.Conv2d(kernel_size=(3, 3), in_channels=in_channels, out_channels=mid_channels, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            nn.Conv2d(kernel_size=(3, 3), in_channels=mid_channels, out_channels=out_channels, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        self.up = nn.ConvTranspose2d(in_channels,in_channels,2,2)

    def forward(self, d, e=None):
        # 上采样
        d = F.interpolate(d, scale_factor=2, mode='bilinear', align_corners=True)
        # 拼接
        if e is not None:
            cat = torch.cat([e, d], dim=1)
            out = self.block(cat)
        else:
            out = self.block(d)
        return out

# 定义最后一层卷积块
class final_block(nn.Module):
    def __init__(self,in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(kernel_size=(3, 3), in_channels=in_channels, out_channels=out_channels, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self,x):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.block(x)
        return x

def get_resnet34():
    try:
        model = models.resnet34(weights=None)
    except TypeError:
        model = models.resnet34(pretrained=False)
    model.avgpool = nn.Identity()
    model.fc = nn.Identity()
    return model

# 定义 Resnet34_Unet 类
class Resnet34_Unet(nn.Module):
    # 定义初始化函数
    def __init__(self, out_channel):
        # 调用 nn.Module 的初始化函数
        super(Resnet34_Unet, self).__init__()

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

        # 定义 Decode 部分，包括四个 expansive_block 和一个 final_block
        self.conv_decode4 = nor_expansive_block(1024 + 512, 512, 512)
        self.conv_decode3 = expansive_block(512 + 256, 256, 256)
        self.conv_decode2 = expansive_block(256 + 128, 128, 128)
        self.conv_decode1 = expansive_block(128 + 64, 64, 64)
        self.conv_decode0 = expansive_block(64, 32, 32)
        self.final_layer = final_block(32, out_channel)

    # 定义前向传播函数
    def forward(self, x):
        # 执行 layer0
        x = self.layer0(x)
        # 执行 Encode
        encode_block1 = self.layer1(x)
        encode_block2 = self.layer2(encode_block1)
        encode_block3 = self.layer3(encode_block2)
        encode_block4 = self.layer4(encode_block3)

        # 执行 Bottleneck
        bottleneck = self.bottleneck(encode_block4)
        # 执行 Decode
        decode_block4 = self.conv_decode4(bottleneck, encode_block4)
        decode_block3 = self.conv_decode3(decode_block4, encode_block3)
        decode_block2 = self.conv_decode2(decode_block3, encode_block2)
        decode_block1 = self.conv_decode1(decode_block2, encode_block1)
        decode_block0 = self.conv_decode0(decode_block1)
        final_layer = self.final_layer(decode_block0)
        return final_layer


if __name__ == '__main__':
    image = torch.rand(8,3,224,224)
    model = Resnet34_Unet(1)
    output = model(image)
    print(output.shape)
