# camera-ready

import torch
import torch.nn as nn
import torch.nn.functional as F

from Model.DeepLabV3.resnet_atrous import Atrous_resnet50_os16
from Model.DeepLabV3.aspp import ASPP, ASPP_Bottleneck

def depthwise_separable_conv(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False):
    """构造一个深度可分离卷积模块"""
    return nn.Sequential(
        nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels, bias=bias),
        nn.BatchNorm2d(in_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )
def replace_conv_with_depthwise_separable(model):
    for name, module in model.named_children():
        if isinstance(module, nn.Conv2d):
            # 计算卷积层的参数
            in_channels = module.in_channels
            out_channels = module.out_channels
            kernel_size = module.kernel_size[0]  # 假设kernel_size是正方形
            stride = module.stride[0]
            padding = module.padding[0]
            dilation = module.dilation[0]
            bias = module.bias is not None

            # 创建深度可分离卷积层
            depthwise_separable = depthwise_separable_conv(
                in_channels, out_channels, kernel_size, stride, padding, dilation, bias)

            # 替换原有卷积层
            setattr(model, name, depthwise_separable)
        else:
            # 递归替换子模块中的卷积层
            replace_conv_with_depthwise_separable(module)

def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    # print('init successfully')

class DeepLabV3(nn.Module):
    def __init__(self):
        super(DeepLabV3, self).__init__()

        self.num_classes = 2

        self.resnet = Atrous_resnet50_os16() # NOTE! specify the type of ResNet here
        self.aspp = ASPP_Bottleneck() # NOTE! if you use ResNet50-152, set self.aspp = ASPP_Bottleneck(num_classes=self.num_classes) instead
        self.final = nn.Sequential(
            nn.Conv2d(256,256,3,1,1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256,self.num_classes,1)
        )
        self.apply(init_weights)

    def forward(self, x):
        # (x has shape (batch_size, 3, h, w))

        h = x.size()[2]
        w = x.size()[3]

        feature_map = self.resnet(x) # (shape: (batch_size, 512, h/16, w/16)) (assuming self.resnet is ResNet18_OS16 or ResNet34_OS16. If self.resnet is ResNet18_OS8 or ResNet34_OS8, it will be (batch_size, 512, h/8, w/8). If self.resnet is ResNet50-152, it will be (batch_size, 4*512, h/16, w/16))

        output = self.aspp(feature_map) # (shape: (batch_size, num_classes, h/16, w/16))

        output = self.final(output)

        output = F.interpolate(output, size=(h, w), mode="bilinear") # (shape: (batch_size, num_classes, h, w))

        return output

class DeeplabV3plus(nn.Module):
    def __init__(self,dropout_rate = 0.55):
        super(DeeplabV3plus, self).__init__()
        self.dropout_rate = dropout_rate
        self.num_classes = 1

        self.resnet = Atrous_resnet50_os16() # NOTE! specify the type of ResNet here
        self.aspp = ASPP_Bottleneck(dropout=self.dropout_rate) # NOTE! if you use ResNet50-152, set self.aspp = ASPP_Bottleneck(num_classes=self.num_classes) instead
        self.project = nn.Sequential(
            nn.Conv2d(256,48,1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )

        self.classifier = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, self.num_classes, 1)
        )
        self.apply(init_weights)
        # self.apply(replace_conv_with_depthwise_separable)

    def forward(self, x):
        # (x has shape (batch_size, 3, h, w))

        h = x.size()[2]
        w = x.size()[3]

        tuple_output = self.resnet(x)
        low_feature,feature_map = tuple_output[0],tuple_output[1] # low bs,256,64,64 ;fm bs,2048,16,16
        low_feature = self.project(low_feature) #low那里的1*1
        output = self.aspp(feature_map) # (shape: (batch_size, num_classes, h/16, w/16))
        output = F.interpolate(output,size=low_feature.shape[2:],mode='bilinear') # 深层特征图上采样
        output = self.classifier(torch.cat([low_feature,output],1))
        return F.interpolate(output,(h,w),mode="bilinear")


if __name__ == '__main__':
    model = DeeplabV3plus()  # 假设这是您的模型实例
    print(model)
    replace_conv_with_depthwise_separable(model)

    dummy_input = torch.randn(8, 3, 256, 256)

    # 执行前向传播
    output = model(dummy_input)

    # 打印输出尺寸
    print(output.shape)