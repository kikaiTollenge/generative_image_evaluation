import timm
import torch
import torch.nn as nn
from Model.DeepLabV3.resnet_atrous import Atrous_resnet50_os16
from Model.DeepLabV3.deeplabv3 import replace_conv_with_depthwise_separable,depthwise_separable_conv
import numpy as np
from Model.Backbone import resnet50
class SiameseNet(nn.Module):
    def __init__(self,feature_extractor,output_size):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.classifier = nn.Sequential(
            nn.Linear(2 * np.prod(output_size),512),
            nn.ReLU(inplace=True),
            nn.Linear(512,4)
        )
        # self.feature_extractor.apply(replace_conv_with_depthwise_separable)

    def forward(self,image1,image2):
        image1 = self.feature_extractor(image1)[-1]
        image2 = self.feature_extractor(image2)[-1]
        feature_combined = torch.cat([image1, image2], dim=1)
        feature_combined = torch.flatten(feature_combined,1)
        result = self.classifier(feature_combined)
        return result

if __name__ == '__main__':
    # 创建模型实例
    proxy_feature_extractors = {
        'Atrous_resnet50_os16':(2048,16,16),
        'resnet50':(2048,8,8),
    }
    feature_extractor = 'Atrous_resnet50_os16'
    feature_extractor_output_size = proxy_feature_extractors[feature_extractor]
    model = SiameseNet(feature_extractor=Atrous_resnet50_os16(),output_size=feature_extractor_output_size)

    image1 = torch.randn(1, 3, 256, 256)
    image2 = torch.randn(1, 3, 256, 256)

    # 计算模型的输出
    output = model(image1, image2)

    # 打印输出张量的形状
    print(output.shape)  # 应该是 [1, 4]，假设批次大小是1
