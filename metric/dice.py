import torch
import torch.nn as nn

class Dice_conf(nn.Module):
    def __init__(self):
        super(Dice_conf, self).__init__()

    def forward(self, logits, targets):
        epsilon = 1e-6
        logits = torch.sigmoid(logits)
        pred = (logits >= 0.5).float()
        intersection = 2 * torch.sum(pred * targets) + epsilon
        union = torch.sum(pred) + torch.sum(targets) + epsilon
        return intersection/union

if __name__ == '__main__':
    # 创建一个简单的二维数据集和目标，用于测试
    N = 4  # 批次大小
    H, W = 5, 5  # 图像高度和宽度

    # 随机生成预测结果和目标，模拟二元分割任务
    predict = torch.rand(N, 2, H, W)  # 预测结果，值在[0, 1]之间
    target = torch.randint(0, 2, (N, 2, H, W))  # 目标，值为0或1

    # predict_pro = (predict > 0.5)
    # 实例化BinaryDiceLoss对象
    dice_conf = Dice_conf()

    # 计算Dice conf
    conf = dice_conf(predict, target)

    print("Dice conf:", conf.item())
