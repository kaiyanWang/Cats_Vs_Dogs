import torch
from torch.utils.data import DataLoader
import random
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms,datasets
from torch.autograd import Variable
import numpy as np
from torchvision import models
from net.ConvNet import ConvNet
from net.FullyConnectedNet import FullyConnectedNet
from options import Options
from mydataset import DogCatDataSet
import matplotlib.pyplot as plt

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed(seed)  # gpu
    torch.cuda.manual_seed_all(seed)  # all gpus

use_gpu = torch.cuda.is_available()

# 定义训练过程
def train(train_loader):
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    model.train()

    for i, data in enumerate(train_loader, 0):
        inputs, train_labels = data
        if use_gpu:
            inputs, labels = Variable(inputs.cuda()), Variable(train_labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(train_labels)
        # inputs, labels = Variable(inputs), Variable(train_labels)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        train_loss += loss.item()
        _, train_predicted = torch.max(outputs.data, 1)
        train_correct += (train_predicted == labels.data).sum()
        train_total += train_labels.size(0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    loss = train_loss / len(train_loader)
    acc = 100 * train_correct / train_total
    return loss, acc

def test(test_loader):
    test_loss = 0.0
    correct = 0
    test_total = 0

    model.eval()

    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            if use_gpu:
                images, labels = Variable(images.cuda()), Variable(labels.cuda())
            else:
                images, labels = Variable(images), Variable(labels)
            outputs = model(images)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels.data).sum()
            test_total += labels.size(0)

    loss = test_loss / len(test_loader)
    acc = 100 * correct / test_total
    return loss, acc

# create loss_fig, psnr_fig, ssim_fig
def create_figs(start_epoch, epoch, train_losses, train_accs, test_losses, test_accs):
    # 将可能的GPU张量转换为CPU上的NumPy数组
    train_accs = [x.cpu().numpy() if isinstance(x, torch.Tensor) else x for x in train_accs]
    test_accs = [x.cpu().numpy() if isinstance(x, torch.Tensor) else x for x in test_accs]
    train_losses = [x.cpu().numpy() if isinstance(x, torch.Tensor) else x for x in train_losses]
    test_losses = [x.cpu().numpy() if isinstance(x, torch.Tensor) else x for x in test_losses]

    plt.switch_backend('Agg')
    plt.figure(1)
    # 绘制训练和验证准确率图
    plt.plot(range(start_epoch, epoch + 1), train_accs, 'b--o', label='train_accuracy')
    plt.plot(range(start_epoch, epoch + 1), test_accs, 'r-o', label='val_accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.pause(0.01)
    plt.savefig('./accuracy.png')
    plt.close()

    plt.figure(2)
    # 绘制训练和验证损失图
    plt.plot(range(start_epoch, epoch + 1), train_losses, 'b--o', label='train_loss')
    plt.plot(range(start_epoch, epoch + 1), test_losses, 'r-o', label='val_loss')
    plt.xlabel('Epochs')
    plt.ylabel('loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.pause(0.01)
    plt.savefig('./loss.png')
    plt.close()

def select_model():
    if opt.Model == "FullyConnectedNet":
        model = FullyConnectedNet()         #  使用全连接网络
    elif opt.Model == "ConvNet":
        model = ConvNet()                 #  使用CNN
    elif opt.Model == "TransLearning":
        # 使用ResNet18预训练模型
        model = models.resnet18(pretrained=True)
        # 冻结所有参数
        for param in model.parameters():
            param.requires_grad = False
        # 修改最后一层
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)  # 改为2分类
    elif opt.Model == "TransLearning_adjust":
        # 使用ResNet18预训练模型
        model = models.resnet18(pretrained=True)
        # 冻结所有参数
        for param in model.parameters():
            param.requires_grad = False
        # 解冻最后的残差块layer4
        for param in model.layer4.parameters():
            param.requires_grad = True
        # 修改最后一层
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(opt.Dropout),  # 添加Dropout减少过拟合
            nn.Linear(num_ftrs, 2)
        )

    return model

def select_data_transform():
    if opt.Model == "FullyConnectedNet":
        # 数据预处理
        data_transform = transforms.Compose([
            transforms.Resize((150, 150)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    else:
        data_transform = transforms.Compose([
            transforms.Resize((224, 224)),  # 调整为224x224
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet归一化参数
        ])
    return data_transform


if __name__ == '__main__':
    opt = Options()
    set_seed(1)

    # Model
    model = select_model()
    if use_gpu:
        model.cuda()

    # Loss
    criterion = nn.CrossEntropyLoss()

    # optimizer
    optimizer = optim.SGD(model.parameters(), lr=opt.Learning_Rate, momentum=0.9)
    if opt.Model == "TransLearning_adjust":
        # 模型微调
        optimizer = optim.Adam([
            {'params': model.layer4.parameters(), 'lr': 1e-5},
            {'params': model.fc.parameters(), 'lr': 1e-3}
        ])
    elif opt.Model == "TransLearning":
        optimizer = optim.Adam(model.fc.parameters(), lr=0.001)  # 只优化最后一层

    # scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # # 加载数据集
    # train_dataset = datasets.ImageFolder(opt.Train_PATH, transform=select_data_transform())
    # test_dataset = datasets.ImageFolder(opt.Test_PATH, transform=select_data_transform())
    #
    # # 创建数据加载器
    # train_loader = DataLoader(train_dataset, batch_size=opt.Batch_Size, shuffle=True)
    # test_loader = DataLoader(test_dataset, batch_size=opt.Batch_Size)

    #  自定义DogCatDataSet
    # 加载数据集
    train_dataset = DogCatDataSet(img_dir=opt.Train_PATH, transform=select_data_transform())
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.Batch_Size, shuffle=True, num_workers=8)

    # 创建数据加载器
    test_dataset = DogCatDataSet(img_dir=opt.Test_PATH, transform=select_data_transform())
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.Batch_Size, shuffle=True, num_workers=8)

    start_epoch = 1
    best_acc = 0
    train_losses = []  # 记录train集的损失
    train_accs = []  # 记录train集的acc
    test_losses = []  # 记录test集的损失
    test_accs = []  # 记录test集的acc

    for epoch in range(start_epoch, opt.NUM_EPOCHS+1):
        # 1.Train
        train_loss, train_acc = train(train_loader)
        print(f'Epoch {epoch}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        train_losses.append(train_loss)  # 记录train集的损失
        train_accs.append(train_acc)  # 记录train集的acc

        # 2.Eval
        test_loss , test_acc = test(test_loader)
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), 'model.pth')
        print('Test set: Test Loss: {:.4f}, Test Acc: {:.2f}%\n'.format(test_loss, test_acc))
        test_losses.append(test_loss)  # 记录test集的损失
        test_accs.append(test_acc)  # 记录test集的acc
        # 画图
        create_figs(start_epoch, epoch, train_losses, train_accs, test_losses, test_accs)

        scheduler.step()

    print("best_acc:",best_acc)