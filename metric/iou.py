import torch
import torch.nn as nn

class IOU(nn.Module):
    def __init__(self,num_class=1):
        super().__init__()

    def forward(self, logits, targets):
        epsilon = 1e-6
        total_iou = 0.0
        batch_size = logits.size(0)

        # 将logits通过sigmoid转换为概率，然后应用阈值获得二进制预测
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()  # 预测标签

        # 计算每个批次的TP, FP, FN
        for i in range(batch_size):
            # 真实标签转换为浮点数以进行计算
            true_labels = targets[i].float()

            # 计算交集和并集
            intersection = torch.sum(preds[i] * true_labels)
            union = torch.sum(preds[i]) + torch.sum(true_labels) - intersection

            # 避免除以零
            union = torch.clamp(union, min=epsilon)

            # 计算IoU
            iou = intersection / union
            total_iou += iou

        # 计算mIoU作为整个批次的平均IoU
        m_iou = total_iou / batch_size

        return m_iou


if __name__ == '__main__':
    mIoU_criterion = IOU()
    output = torch.randn(8, 1, 256, 256)
    labels = torch.randint(0, 2, (8, 256, 256))
    iou_score = mIoU_criterion(output, labels)
    print(f"IoU score: {iou_score}")