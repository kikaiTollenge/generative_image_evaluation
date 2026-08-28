import torch.optim as optim
import torch
import torch.nn as nn
import torchvision.transforms.functional as F
import random
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from Model.Siamesenet import SiameseNet
from Dataset.dataset import PathologyData,glas_classify_Dataset
from tqdm import tqdm
from options import args
from Model.Backbone import resnet50, deeplabv3plusencoder
from metric.dice import Dice_conf
from metric.iou import IOU
from Model.DeepLabV3.deeplabv3 import DeeplabV3plus
import numpy as np
from Model.DeepLabV3.resnet_atrous import Atrous_resnet50_os16
import multiprocessing as mp
from Model.Unet.unet import UNet
from Model.Unet.resunet34 import Resnet34_Unet
from Model.Unet.resunet import ResUnet
from Model.Unet.resunet_plus import ResUnetPlusPlus
from Dataset.dataset import show_image_and_label
from Model.classify_model import Res34_encoder,Res34_pruningencoder
import os
device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')

def load_checkpoint(model,path):
    checkpoint = torch.load(path)
    state_dict = checkpoint["state_dict"]
    backbone_weight_dict = {}
    for key in state_dict.keys():
        if key.startswith('module.'):
            clean_key = key[len('module.'):]

        if "encoder_q.feature_extractor.resnet" in clean_key:
            layer_key = clean_key.split("encoder_q.feature_extractor.")[1]
            backbone_weight_dict[layer_key] = state_dict[key]

        if "encoder_q.feature_extractor.layer" in clean_key:
            layer_key = clean_key.split("encoder_q.feature_extractor.")[1]
            # 更新键，使其与 moco_backbone 中的名称相匹配
            backbone_weight_dict[layer_key] = state_dict[key]

        if "encoder_q.feature_extractor.bottleneck" in clean_key:
            layer_key = clean_key.split('encoder_q.feature_extractor.')[1]
            backbone_weight_dict[layer_key] = state_dict[key]

    model.load_state_dict(backbone_weight_dict,strict=False)
    for name, param in model.named_parameters():
        if name in backbone_weight_dict:
            param.requires_grad = True
    print('加载成功')
    return model

def random_checkpoint(model):
    backbone_weight_dict = {}

    # Identify the specific layers for random initialization and gradient blocking
    for name, param in model.named_parameters():
        if "encoder_q.feature_extractor.resnet" in name \
                or "encoder_q.feature_extractor.layer" in name \
                or "encoder_q.feature_extractor.bottleneck" in name:
            backbone_weight_dict[name] = param

    # Random initialization and blocking gradients for identified layers
    for name, param in backbone_weight_dict.items():
        with torch.no_grad():  # Ensure no gradient calculation
            if param.requires_grad:
                if len(param.shape) > 1:  # Convolutional or Linear layer
                    if param.dim() > 1:
                        nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
                    else:  # Linear layer
                        nn.init.normal_(param, mean=0.0, std=0.02)
                elif 'bias' in name:  # Bias term
                    nn.init.constant_(param, 0)
                else:  # BatchNorm layer
                    nn.init.constant_(param, 1)

    # Block gradients for the identified layers
    for name, param in model.named_parameters():
        if name in backbone_weight_dict:
            param.requires_grad = False

    print('随机初始化成功')
    return model


def l1_regularization_loss(model,lambda_l1):
    return lambda_l1 * sum(p.abs().sum() for p in model.parameters())

def replace_backbone(name):
    # backbone raplace area
    proxy_feature_extractors = {
        'Atrous_resnet50_os16':(2048,14,14),
        'resnet50':(2048,8,8),
        'deeplabv3plusencoder':(256,16,16)
    }
    feature_extractor_output_size = proxy_feature_extractors[name] # get output's shape
    if name == 'Atrous_resnet50_os16':
        feature_extractor_instance = Atrous_resnet50_os16()
    if name == 'resnet50':
        feature_extractor_instance = resnet50()
    if name == 'deeplabv3plusencoder':
        feature_extractor_instance = deeplabv3plusencoder()
    return feature_extractor_instance,feature_extractor_output_size
    # backbone raplace area

# 动态学习率
def poly_lr_scheduler(optimizer ,init_lr,max_iter,power = 0.9):
    def lambda_epoch(epoch):
        factor = (1 - epoch / max_iter) ** power
        return factor
    return torch.optim.lr_scheduler.LambdaLR(optimizer,lr_lambda=lambda_epoch)

def visualize_sample(image, label, output, epoch, index):
    """可视化单个样本的输入、标签和输出，显示3秒后自动关闭窗口"""
    image_np = show_image_and_label(image.detach().cpu())
    label_np = label[0].detach().cpu().numpy()
    output_np = output[0].detach().cpu().numpy()

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(image_np)
    axs[0].set_title('Input Image')
    axs[1].imshow(label_np, cmap='gray')
    axs[1].set_title('True Label')
    axs[2].imshow(output_np, cmap='gray')
    axs[2].set_title('Model Output')
    plt.suptitle(f'Epoch: {epoch+1}, Sample: {index+1}')

    plt.show(block=False)  # 非阻塞显示
    plt.pause(3)  # 显示3秒
    plt.close()  # 关闭窗口

def rotate_predict_test(model, dataloader,weight_path=None):
    if weight_path is not None:
        model.load_state_dict(torch.load(weight_path))  # 加载指定的权重
    model.eval()  # 将模型设置为评估模式
    correct = 0
    total = 0

    with torch.no_grad():  # 在不计算梯度的情况下执行前向传递
        for image in dataloader:
            image = image.to(device)
            image_rotated = torch.zeros_like(image)
            rotation_labels = torch.zeros(image.size(0), dtype=torch.long, device=device)
            for i in range(image.size(0)):
                factor = random.randint(0, 3)
                image_rotated[i] = F.rotate(image[i], factor * 90)
                rotation_labels[i] = factor
            image_rotated = image_rotated.to(device)

            outputs = model(image, image_rotated)
            _, predicted = torch.max(outputs.data, 1)
            total += rotation_labels.size(0)
            correct += (predicted == rotation_labels).sum().item()
    return 100 * correct / total

def rotate_predict_train(type,model, train_dataloader,test_dataloader, criterion, optimizer, num_epochs = 100):
    if type == 'True_image':
        torch.save(model.state_dict(), './weight/rotation/proxy_model_initial.pth')  # 训练真实图片时先保存权重
    if type == 'Fake_image':
        model.load_state_dict(torch.load('./weight/rotation/proxy_model_initial.pth'))  # 训练生成图片时加载训练真实图片时的初始权重
    model.train()
    losses = []  # 存储每个epoch的loss

    best_accuracy = 0  # 初始化最高准确率
    best_accuracy_epoch = -1  # 初始化最高准确率对应的epoch

    for epoch in tqdm(range(num_epochs)):
        model.train()
        running_loss = 0.0

        for image in train_dataloader:
            optimizer.zero_grad()  # 梯度清零
            image = image.to(device)
            image_rotated = torch.zeros_like(image)  # 旋转后的图像存入x_rotated
            rotation_labels = torch.zeros(image.size(0), dtype=torch.long,device=device)  # 生成torch来存储对应的标签,并存到gpu上
            for i in range(image.size(0)):  # 为batch里每个图像设置不同的rotate
                factor = random.randint(0, 3)
                image_rotated[i] = F.rotate(image[i], factor * 90)
                rotation_labels[i] = factor
            image_rotated = image_rotated.to(device)  # 旋转后的图像存到GPU上

            outputs = model(image, image_rotated)  # 前向
            loss = criterion(outputs, rotation_labels)  # 求loss
            loss.backward()  # 反向传播
            optimizer.step()  # 权重更新

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_dataloader)
        losses.append(epoch_loss)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss}")

        # 绘制并保存loss曲线图
        plt.figure()
        plt.plot(losses)
        plt.title('Loss Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig(f'./weight/rotation/{type}_image_loss_curve.png')
        plt.close()

        # test
        accuracy = rotate_predict_test(model,test_dataloader)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_accuracy_epoch = epoch + 1
            torch.save(model.state_dict(), f'./weight/rotation/{type}_model_min_loss.pth')
            print(f'New best model saved with accuracy: {best_accuracy}% at epoch {best_accuracy_epoch}')

    print(f'Train completed.Best model saved with accuracy: {best_accuracy}% at epoch {best_accuracy_epoch}')

def full_supervision_train(model , train_dataloader, test_dataloader,criterion ,optimizer , num_epochs = 100, path = 'full_supervision'):
    scheduler = poly_lr_scheduler(optimizer, init_lr=0.007, max_iter=num_epochs,power=0.9)
    losses = []  # 存储每个epoch的loss
    conf_best = 0.0
    min_loss = float('inf')  # 初始化最小loss为无穷大
    min_loss_epoch = -1  # 初始化最小loss对应的epoch

    for epoch in tqdm(range(num_epochs), desc='Training'):
        running_loss = 0.0
        model.train()
        for i, (image, label) in enumerate(train_dataloader):
            l1loss = l1_regularization_loss(model,lambda_l1=0.01)
            optimizer.zero_grad()  # 梯度清零
            output = model(image)  # 前向
            loss = criterion(output, label) + l1loss # 求loss logit值输入outputs
            loss.backward()  # 反向传播
            optimizer.step()  # 权重更新

            running_loss += loss.item()

            # if i == 0:  # 每个epoch观测第一个batch的第一个样本
            #     with torch.no_grad():
            #         model.eval()
            #         output = output[:1]  # 假设你只想观测第一个样本
            #         visualize_sample(image[0], label[0], output[0], epoch, i)
            #         model.train()

        scheduler.step()

        epoch_loss = running_loss / len(train_dataloader)
        losses.append(epoch_loss)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss}")

        # 绘制并保存loss曲线图
        plt.figure()
        plt.plot(losses)
        plt.title('Loss Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig(f'./weight/{path}/{path}_loss_curve.png')
        plt.close()

        # 检查并更新最大Dice系数及其对应的epoch
        dice,iou,test_loss = supervision_test(model,test_dataloader)
        print("Test loss: ",test_loss)
        if iou > conf_best:
            conf_best = iou
            max_conf = round(iou,4)
            max_conf_epoch = epoch + 1
            torch.save(model.state_dict(),
                       f'weight/{path}/{path}_model_min_loss.pth')  # 保存最小loss对应的模型
            print(f"Max iou is {max_conf} at Epoch {max_conf_epoch},dice is {dice},loss is {test_loss}")

def full_supervision_test(model , dataloader,path = 'full_supervision'):
    model.load_state_dict(torch.load(f'./weight/{path}/{path}_model_min_loss.pth'))
    model.eval()  # 将模型设置为评估模式
    total_loss = 0  # 用于累积所有批次的损失
    with torch.no_grad():
        for image,labels in dataloader:
            image = image.to(device)
            labels = labels.to(device)
            labels = F.sig  # 只取第一个通道的值，因为三个通道值相同
            output = model(image)  # 获取模型输出
            # loss = iou_metric(output, labels)
            # total_loss += (-1 * loss + 1)

            # 将图像、输出和标签从GPU移动到CPU，并转换为NumPy数组
            image_np = image[0].cpu().numpy().transpose(1, 2, 0)
            output_np = torch.sigmoid(output[0]).cpu().numpy().argmax(0)
            labels_np = labels[0].cpu().numpy()
            # 逆标准化处理
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image_np = (image_np * std + mean) * 255
            image_np = np.clip(image_np, 0, 255).astype(np.uint8)

            # 保存图像、输出和标签
            plt.imsave('image.png', image_np)
            plt.imsave('output.png', output_np, cmap='gray')
            plt.imsave('label.png', labels_np, cmap='gray')

    avg_loss = total_loss / len(dataloader)  # 计算平均系数
    print(f'Average IOU Score: {avg_loss}')  # 打印平均Dice系数

def deeplab_downstream_test(model, dataloader, weight_path=None):
    if weight_path is not None:
        model.load_state_dict(torch.load(weight_path))  # 加载指定的权重
    model.eval()  # 将模型设置为评估模式
    dice_loss = Dice_conf()  # 初始化Dice损失函数
    total_loss = 0  # 用于累积所有批次的损失
    with torch.no_grad():
        for image, labels in dataloader:
            image = image.to(device)
            labels = labels.to(device)
            labels = labels[:, 0, :, :]  # 只取第一个通道的值
            output = model(image)  # 获取模型输出
            loss = dice_loss(output, labels)  # 计算Dice损失
            total_loss += (-1 * loss.item() + 1)  # 计算Dice系数

            # # 将图像、输出和标签从GPU移动到CPU，并转换为NumPy数组
            # image_np = image[1].cpu().numpy().transpose(1, 2, 0)
            # output_np = torch.sigmoid(output[1]).cpu().numpy().argmax(0)
            # labels_np = labels[1].cpu().numpy()
            # # 逆标准化处理
            # mean = np.array([0.485, 0.456, 0.406])
            # std = np.array([0.229, 0.224, 0.225])
            # image_np = (image_np * std + mean) * 255
            # image_np = np.clip(image_np, 0, 255).astype(np.uint8)
            #
            # # 保存图像、输出和标签
            # plt.imsave('image.png', image_np)
            # plt.imsave('output.png', output_np, cmap='gray')
            # plt.imsave('label.png', labels_np, cmap='gray')

    avg_loss = total_loss / len(dataloader)  # 计算平均Dice系数
    print(f'Average Dice Coefficient: {avg_loss}')  # 打印平均Dice系数
    return avg_loss  # 返回平均Dice系数

def deeplab_downstream_train(type, model, train_dataloader, test_dataloader, criterion, optimizer, lr,dp,num_epochs=100):
    scheduler = poly_lr_scheduler(optimizer, init_lr=lr, max_iter=num_epochs, power=0.9)
    if type == 'True_image':
        torch.save(model.state_dict(), './weight/downstream_deeplab/downstream_model_initial.pth')
    if type == 'Fake_image':
        model.load_state_dict(torch.load('./weight/downstream_deeplab/downstream_model_initial.pth'))
    model.resnet.load_state_dict(torch.load(f'./weight/rotation/{type}_model_min_loss.pth'), strict=False)
    for param in model.resnet.parameters():
        param.requires_grad = False
    losses = []
    max_dice_epoch = -1
    max_dice = 0  # 初始化最大Dice系数

    for epoch in tqdm(range(num_epochs), desc='Training'):
        model.train()
        running_loss = 0.0
        for image, labels in train_dataloader:
            optimizer.zero_grad()
            image = image.to(device)
            labels = labels.to(device)
            labels = torch.squeeze(labels, dim=1)
            labels = labels.long()
            outputs = model(image)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        scheduler.step()
        # current_lr = optimizer.param_groups[0]['lr']
        # print(f"Epoch{epoch+1}/{num_epochs},current lr is {current_lr}")
        epoch_loss = running_loss / len(train_dataloader)
        losses.append(epoch_loss)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss}")

        # 测试逻辑
        avg_dice = deeplab_downstream_test(model, test_dataloader)  # 调用测试函数
        if avg_dice > max_dice:
            max_dice = round(avg_dice,4)
            max_dice_epoch = epoch+1
            torch.save(model.state_dict(), f'./weight/downstream_deeplab/{type}_deeplab_downstream_model_max_dice.pth')  # 保存Dice系数最高的模型
            print(f'Max dice of {max_dice} at epoch {max_dice_epoch}')

    plt.figure()
    plt.plot(losses)
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(f'./weight/downstream_deeplab/{type}_deeplab_downstream_loss_curve.png')
    plt.close()
    with open ('try_param','+a')as file:
        file.write(f"--------学习率为{lr},dropout率为{dp},最大dice为{max_dice},在{max_dice_epoch}---------\n")

    # if (epoch + 1) % 10 == 0:
    #     torch.save(model.state_dict(), f'weight/downstream_deeplab/{type}_model_epoch_{epoch + 1}.pth')


    print(f"Training complete. Max dice of {max_dice} at epoch {max_dice_epoch}")

def unet_supervision_train(model , train_dataloader, test_dataloader,criterion ,optimizer , num_epochs = 100, path = 'full_supervision',type='Fake_image'):
    losses = []
    min_loss = float('inf')
    min_loss_epoch = -1
    conf_best = 0.0
    for epoch in tqdm(range(num_epochs),desc="Training"):
        running_loss = 0.0
        model.train()
        for i,(image,label) in enumerate(train_dataloader):
            optimizer.zero_grad()
            image = image.to(device)
            label = label.to(device)
            output = model(image)
            loss = criterion(output,label)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # if i == 0:  # 每个epoch观测第一个batch的第一个样本
            #     with torch.no_grad():
            #         model.eval()
            #         output = output[:1]  # 假设你只想观测第一个样本
            #         visualize_sample(image[0], label[0], output[0], epoch, i)
            #         model.train()

        epoch_loss = running_loss / len(train_dataloader)
        losses.append(epoch_loss)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss}")

        valid_loss = []
        dice,iou,classifier_loss = supervision_test(model,test_dataloader,type=type)
        valid_loss.append(classifier_loss)


        plt.figure(figsize=(10, 5))
        plt.plot(losses, label='Train Loss')  # 添加标签用于图例
        plt.plot(valid_loss, label='Valid Loss', linestyle='--', color='r')  # 添加标签和不同的线型与颜色
        plt.title('Loss Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()  # 显示图例
        plt.savefig(f'./weight/{path}/{type}_{path}_segmentation_loss_curve.png')
        plt.close()

        if iou > conf_best:
            conf_best = iou
            max_conf = round(iou,4)
            max_conf_epoch = epoch + 1
            torch.save(model.state_dict(),
                       f'weight/{path}/{type}_{path}_segmentation_best_model.pth')  # 保存最小loss对应的模型
            print(f"Max conf is {max_conf} at Epoch {max_conf_epoch}, dice is {dice},loss is {classifier_loss}")

        # print(f"Training complete. Lowest loss of {min_loss} at epoch {min_loss_epoch}")

def supervision_test(model, test_dataloader, weight_path = None,type ='Fake_image'):
    if weight_path is not None:
        model.load_state_dict(torch.load(f'./weight/{weight_path}/{type}_{weight_path}_segmentation_best_model.pth'))
    running_loss = 0.0
    running_dice = 0.0
    running_iou = 0.0

    iou_conf = IOU()
    dice_conf = Dice_conf()
    loss = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        model.eval()
        for k,(image,label) in enumerate(test_dataloader):
            image = image.to(device)
            label = label.to(device)
            output = model(image)
            total_loss = loss(output,label)

            # 应用 softmax 获取每个类别的概率
            probabilities = torch.sigmoid(output)

            # 获取概率最大的类别索引
            _, predicted_classes = torch.max(probabilities, dim=1)


            iou = iou_conf(predicted_classes,label)
            iou = iou.cpu().item()
            dice = dice_conf(predicted_classes,label)
            dice = dice.cpu().item()

            running_iou += iou
            running_dice += dice
            running_loss += total_loss.item()


            # if k == 1:
            #     plt.ioff()
            #     for i in range(3):  # 显示前3张图像
            #         plt.figure(figsize=(10, 5))
            #
            #         plt.subplot(1,3,1)
            #         plt.imshow(image[i].cpu().permute(1,2,0))
            #         plt.title('Image')
            #
            #         plt.subplot(1, 3, 2)
            #         plt.imshow(binary_output[i].cpu())
            #         plt.title('Output')
            #
            #         plt.subplot(1, 3, 3)
            #         plt.imshow(label[i].cpu().permute(1,2,0))
            #         plt.title('Label')
            #
            #         plt.savefig(f'unet_{i}.png')

        # torch.cuda.empty_cache()
        classifier_loss = running_loss / len(test_dataloader)
        dice = running_dice / len(test_dataloader)
        iou = running_iou / len(test_dataloader)
        # print(f"Test Loss: {classifier_loss}")
        # print(f"Conf: {conf_loss}")
        return dice,iou,classifier_loss

def glas_grade_train(model , train_dataloader, test_dataloader,criterion ,optimizer , num_epochs = 50, path = 'grade',binary = True,type = 'Fake_image',weight = 0.01):
    classType = ' '
    if binary:
        classType = '2_class'   # 这里原来是two_class 改一下名，防止把原来的给覆盖了
    else:
        classType = 'five_class'

    losses = []
    valid_loss = []
    test_accuracy_list = []
    train_accuracy_list = []
    correct = 0
    total = 0

    conf_best = 0.0
    for epoch in tqdm(range(num_epochs),desc="Training"):
        running_loss = 0.0
        model.train()
        for i,(image,label) in enumerate(train_dataloader):
            image = image.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            output = model(image)
            l1loss = l1_regularization_loss(model,lambda_l1=weight)
            loss = criterion(output,label)+l1loss
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            _, predicted = torch.max(output, 1)
            correct += (predicted == label).sum().item()
            total += label.size(0)

        epoch_loss = running_loss / len(train_dataloader)
        losses.append(epoch_loss)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss}")
        train_accuracy = correct/total

        dir_path = f'./weight/{path}'
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)


        classifier_loss,test_accuracy= glas_grade_test(model,test_dataloader)

        valid_loss.append(classifier_loss)

        train_accuracy_list.append(train_accuracy)
        test_accuracy_list.append(test_accuracy)

        plt.figure(figsize=(10, 5))
        plt.plot(losses, label='Train Loss',linestyle='--',color ='g')  # 添加标签用于图例
        plt.plot(valid_loss, label='Valid Loss', linestyle='--', color='r')  # 添加标签和不同的线型与颜色
        plt.title('Loss Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()  # 显示图例
        plt.savefig(f'{dir_path}/{type}_{path}_{classType}_loss_curve.png')
        plt.close()

        plt.figure(figsize=(10, 5))
        plt.plot(train_accuracy_list, label='Train Accuracy',linestyle='--',color ='g')  # 添加标签用于图例
        plt.plot(test_accuracy_list, label='Valid Accuracy', linestyle='--', color='r')  # 添加标签和不同的线型与颜色
        plt.title('Accuracy Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()  # 显示图例
        plt.savefig(f'{dir_path}/{type}_{path}_{classType}_accuracy_curve.png')
        plt.close()

        if test_accuracy > conf_best:
            conf_best = test_accuracy
            max_conf = round(test_accuracy,4)
            max_conf_epoch = epoch + 1
            torch.save(model.state_dict(),
                       f'{dir_path}/{type}_{path}_{classType}_model.pth')  # 保存最小loss对应的模型
            print(f"Max conf is {max_conf} at Epoch {max_conf_epoch},loss is {classifier_loss}")

def glas_grade_test(model, test_dataloader, path = None,binary = True,type = 'Fake_image'):
    classType = ' '
    if binary:
        classType = 'two_class'
    else:
        classType = 'five_class'

    if path is not None:
        dir_path = f'./weight/{path}'
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)
        model.load_state_dict(torch.load(f'{dir_path}/{type}_{path}_{classType}_model.pth'))
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0

    loss = nn.CrossEntropyLoss()

    with torch.no_grad():
        for k,(image,label) in enumerate(test_dataloader):
            image = image.to(device)
            label = label.to(device)
            output = model(image)
            total_loss = loss(output,label)
            running_loss += total_loss.item()

            _, predicted = torch.max(output, 1)
            correct += (predicted == label).sum().item()
            total += label.size(0)

        classifier_loss = running_loss / len(test_dataloader)
        accuracy = correct / total
        return classifier_loss,accuracy


def glas_grade_train_try(model , train_dataloader, test_dataloader,criterion ,optimizer , num_epochs = 50, path = 'grade',binary = True,type = 'Fake_image',weight = 0.01,best_conf=0.0):
    classType = ' '
    if binary:
        classType = '2_class'   # 这里原来是two_class 改一下名，防止把原来的给覆盖了
    else:
        classType = 'five_class'

    losses = []
    valid_loss = []
    test_accuracy_list = []
    train_accuracy_list = []
    correct = 0
    total = 0

    for epoch in tqdm(range(num_epochs),desc="Training"):
        running_loss = 0.0
        model.train()
        for i,(image,label) in enumerate(train_dataloader):
            image = image.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            output = model(image)
            l1loss = l1_regularization_loss(model,lambda_l1=weight)
            loss = criterion(output,label)+l1loss
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            _, predicted = torch.max(output, 1)
            correct += (predicted == label).sum().item()
            total += label.size(0)

        epoch_loss = running_loss / len(train_dataloader)
        losses.append(epoch_loss)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss}")
        train_accuracy = correct/total

        dir_path = f'./weight/{path}'
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)


        classifier_loss,test_accuracy= glas_grade_test(model,test_dataloader)

        valid_loss.append(classifier_loss)

        train_accuracy_list.append(train_accuracy)
        test_accuracy_list.append(test_accuracy)


        if test_accuracy > best_conf:
            best_conf = test_accuracy
            max_conf = round(test_accuracy,4)
            max_conf_epoch = epoch + 1
            torch.save(model.state_dict(),
                       f'{dir_path}/{type}_{path}_{classType}_model.pth')  # 保存最小loss对应的模型
            plt.figure(figsize=(10, 5))
            plt.plot(losses, label='Train Loss',linestyle='--',color ='g')  # 添加标签用于图例
            plt.plot(valid_loss, label='Valid Loss', linestyle='--', color='r')  # 添加标签和不同的线型与颜色
            plt.title('Loss Curve')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()  # 显示图例
            plt.savefig(f'{dir_path}/{type}_{path}_{classType}_loss_curve.png')
            plt.close()

            plt.figure(figsize=(10, 5))
            plt.plot(train_accuracy_list, label='Train Accuracy',linestyle='--',color ='g')  # 添加标签用于图例
            plt.plot(test_accuracy_list, label='Valid Accuracy', linestyle='--', color='r')  # 添加标签和不同的线型与颜色
            plt.title('Accuracy Curve')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()  # 显示图例
            plt.savefig(f'{dir_path}/{type}_{path}_{classType}_accuracy_curve.png')
            plt.close()
            print(f"Max conf is {max_conf} at Epoch {max_conf_epoch},loss is {classifier_loss}")

def glas_grade_test_try(model, test_dataloader, path = None,binary = True,type = 'Fake_image'):
    classType = ' '
    if binary:
        classType = 'two_class'
    else:
        classType = 'five_class'

    if path is not None:
        dir_path = f'./weight/{path}'
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)
        model.load_state_dict(torch.load(f'{dir_path}/{type}_{path}_{classType}_model.pth'))
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0

    loss = nn.CrossEntropyLoss()

    with torch.no_grad():
        for k,(image,label) in enumerate(test_dataloader):
            image = image.to(device)
            label = label.to(device)
            output = model(image)
            total_loss = loss(output,label)
            running_loss += total_loss.item()

            _, predicted = torch.max(output, 1)
            correct += (predicted == label).sum().item()
            total += label.size(0)

        classifier_loss = running_loss / len(test_dataloader)
        accuracy = correct / total
        return classifier_loss,accuracy


if __name__ == '__main__':
    # 'Atrous_resnet50_os16': (2048, 16, 16),
    # 'resnet50': (2048, 8, 8),
    # 'deeplabv3plusencoder': (256, 16, 16)
    mp.set_start_method('spawn')
    feature_extractor_instance,feature_extractor_output_size=replace_backbone('Atrous_resnet50_os16')
    type = 'Fake_image'  # 'True_image' or 'Fake_image'
    trainAndtest = True

    if args.mode == 'downstream_unet':
        train_image_data_path = './Dataset/glas/train/image'
        train_mask_data_path = './Dataset/glas/train/mask'
        test_image_data_path = './Dataset/glas/test/image'
        test_mask_data_path = './Dataset/glas/test/mask'
        trainAndtest = True
        # model = Unet().to(device)
        # model = ResUnetPlusPlus(3).to(device)
        model = Resnet34_Unet(1).to(device)
        model = load_checkpoint(model, path=args.checkpoint) if args.checkpoint else model
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(),lr=0.0001)
        test_dataset = PathologyData(dataset_folder=test_image_data_path,
                                         dataset_label_folder=test_mask_data_path, train=False)
        test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)
        # train
        train_dataset = PathologyData(dataset_folder=train_image_data_path,
                                        dataset_label_folder=train_mask_data_path)
        train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)

        if trainAndtest:
            unet_supervision_train(model, train_dataloader, test_dataloader,criterion=criterion, optimizer=optimizer, num_epochs=100,
                                   path='grade',type=type)
        # test
        supervision_test(model, test_dataloader,weight_path='grade',type = type)

    if args.mode == 'downstream_glas_grade':
        # learning_rate = [1.0, 0.874, 0.523, 0.345, 0.157]
        max = 1e-2
        min = 1e-5
        lr = 5e-5
        step = (max - min) / 9
        # learning_rate = [min + i*step for i in range(10)]
        # learning_rate = [5e-4]
        weight_Decay = []
        for i in weight_Decay:
            binary = False
            model = Res34_pruningencoder(out_channel=5).to(device=device)
            # model = random_checkpoint(model)
            model = load_checkpoint(model, path=args.checkpoint) if args.checkpoint else model
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(),lr = 5e-4,weight_decay=i)
            test_dataset = glas_classify_Dataset(csv_path='./Dataset/glas/Grade_modified.csv',binary=binary,train=False)
            test_dataloader = DataLoader(test_dataset,batch_size=32,shuffle=True)

            if trainAndtest:
                train_dataset = glas_classify_Dataset(csv_path='./Dataset/glas/Grade_modified.csv',binary=binary,train=True)
                train_dataloader = DataLoader(train_dataset,batch_size=32,shuffle=True)
                glas_grade_train(model, train_dataloader, test_dataloader,criterion=criterion, optimizer=optimizer, num_epochs=100,
                                    path='grade',binary=binary,type = type,weight = i)
            classifier_loss,accuracy = glas_grade_test(model, test_dataloader,path='grade',binary = binary,type  = type)
            with open('best_epochs.txt', 'a+') as file:
                file.write(f'学习率为{lr} , 最高准确率为 {accuracy} , loss为 {classifier_loss} \n')


    if args.mode == 'downstream_glas_grade_try':
    # learning_rate = [1.0, 0.874, 0.523, 0.345, 0.157]
        max = 1e-2
        min = 1e-5
        step = (max - min) / 9
        # learning_rate = [min + i*step for i in range(10)]
        # learning_rate = [5e-4]
        l2s = [0.0001, 0.0005, 0.001, 0.005, 0.01]
        l1s = [0.0001, 0.0005, 0.001, 0.01, 0.1]
        learning_rate = [0.00005, 0.00015, 0.00025, 0.00035, 0.00045, 0.00050, 0.00055, 0.00065, 0.00075, 0.00085,0.00049, 0.000495, 0.00050, 0.000505, 0.00051, 0.000515, 0.00052, 0.000525, 0.00053, 0.000535]
        best_conf = 0.0
        for lr in learning_rate:
            for l2 in l2s:
                for l1 in l1s:
                    binary = False
                    model = Res34_pruningencoder(out_channel=5).to(device=device)
                    # model = random_checkpoint(model)
                    model = load_checkpoint(model, path=args.checkpoint) if args.checkpoint else model
                    criterion = nn.CrossEntropyLoss()
                    optimizer = optim.Adam(model.parameters(),lr = lr,weight_decay=l2)
                    test_dataset = glas_classify_Dataset(csv_path='./Dataset/glas/Grade_modified.csv',binary=binary,train=False)
                    test_dataloader = DataLoader(test_dataset,batch_size=32,shuffle=True)

                    if trainAndtest:
                        train_dataset = glas_classify_Dataset(csv_path='./Dataset/glas/Grade_modified.csv',binary=binary,train=True)
                        train_dataloader = DataLoader(train_dataset,batch_size=32,shuffle=True)
                        glas_grade_train_try(model, train_dataloader, test_dataloader,criterion=criterion, optimizer=optimizer, num_epochs=50,
                                            path='grade',binary=binary,type = type,weight = l1,best_conf=best_conf)
                    classifier_loss,accuracy = glas_grade_test_try(model, test_dataloader,path='grade',binary = binary,type  = type)
                    if(accuracy > best_conf):
                        best_conf = accuracy
                    with open('best_epochs.txt', 'a+') as file:
                        file.write(f'学习率为{lr} , 最高准确率为 {accuracy} , loss为 {classifier_loss},l2为{l2},l1为{l1} \n')














    if args.mode == 'rotation_train_resnet':
        if type == 'True_image':
            train_path = './Dataset/true_image/train/images'
            test_path = './Dataset/true_image/test/images'
        if type == 'Fake_image':
            train_path = args.train_path
            test_path = args.test_path
        model = SiameseNet(feature_extractor=feature_extractor_instance,output_size=feature_extractor_output_size).to(device if torch.cuda.is_available() else 'cpu')
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        train_dataset = PathologyData(dataset_folder=train_path)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_dataset = PathologyData(dataset_folder=test_path, train=False)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

        if trainAndtest:
            # train
            rotate_predict_train(type, model, train_dataloader, test_dataloader,criterion, optimizer)
        # test
        _ = rotate_predict_test(model, test_dataloader,weight_path=f'./weight/rotation/{type}_model_min_loss.pth')

    if args.mode == 'rotation_deeplab_downstream':
        # 这里的type主要指用于预训练的图片是什么类型
        # 这里要求先训练真实图片，训练开始前保存模型初始权重，训练生成的图像时会加载之前训练真实图片的权重
        lr = 0.001  # 0.007
        model = DeeplabV3plus()
        model = model.to(device if torch.cuda.is_available() else 'cpu')
        # criterion = SoftDiceLoss()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        train_dataset = PathologyData(dataset_folder=args.downstream_train_data,
                                      dataset_label_folder=args.downstream_train_label)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_dataset = PathologyData(dataset_folder=args.downstream_test_data,
                                         dataset_label_folder=args.downstream_test_label, train=False)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

        if trainAndtest:
            deeplab_downstream_train(type, model, train_dataloader, test_dataloader, criterion=criterion,
                                     optimizer=optimizer,lr=lr,num_epochs=100)

        _ = deeplab_downstream_test(model, test_dataloader,weight_path=f'./weight/downstream_deeplab/{type}_deeplab_downstream_model_max_dice.pth')

    if args.mode == 'full_supervision_deeplab_resnet':
        trainAndtest = True
        model = DeeplabV3plus(dropout_rate=0.6).to(device=device if torch.cuda.is_available() else 'cpu')
        # criterion = SoftDiceLoss()
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(),lr=0.007,weight_decay=1e-4)
        test_dataset = PathologyData(dataset_folder=args.downstream_test_data,
                                         dataset_label_folder=args.downstream_test_label, train=False)
        test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=True, num_workers=0)
        if trainAndtest:
            # train
            train_dataset = PathologyData(dataset_folder=args.downstream_train_data,
                                          dataset_label_folder=args.downstream_train_label)
            train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            full_supervision_train(model, train_dataloader, test_dataloader,criterion=criterion, optimizer=optimizer, num_epochs=100,
                                   path='full_supervision')
        # test
        supervision_test(model, test_dataloader)

    if args.mode == 'try_for_param':
        if type == 'True_image':
            train_path = './Dataset/true_image/train/images'
            test_path = './Dataset/true_image/test/images'
        if type == 'Fake_image':
            train_path = args.train_path
            test_path = args.test_path

        dropout = [0.5258, 0.5973, 0.5733, 0.60599, 0.4832]
        learing_rate = [1.0000, 0.6158, 0.3793, 0.2336, 0.1438, 0.0886, 0.0546, 0.0336, 0.0207, 0.0127, 0.0078, 0.0048, 0.0030, 0.0018, 0.0011, 0.0007, 0.0004, 0.0003, 0.0002, 0.0001]
        for dp in tqdm(dropout):
            for lr in tqdm(learing_rate):
                print(f"现在dp为{dp},lr为{lr}")
                model = DeeplabV3plus(dropout_rate=dp)
                model = model.to(device if torch.cuda.is_available() else 'cpu')
                # criterion = SoftDiceLoss()
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.Adam(model.parameters(), lr=lr)
                train_dataset = PathologyData(dataset_folder=args.downstream_train_data,
                                            dataset_label_folder=args.downstream_train_label)
                train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
                test_dataset = PathologyData(dataset_folder=args.downstream_test_data,
                                                dataset_label_folder=args.downstream_test_label, train=False)
                test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

                if trainAndtest:
                    deeplab_downstream_train(type, model, train_dataloader, test_dataloader, criterion=criterion,
                                            optimizer=optimizer,lr=lr,dp=dp,num_epochs=100)

                _ = deeplab_downstream_test(model, test_dataloader,weight_path=f'./weight/downstream_deeplab/{type}_deeplab_downstream_model_max_dice.pth')
