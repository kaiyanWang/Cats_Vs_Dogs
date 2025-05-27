import torch.nn as nn

# 定义全连接神经网络模型
class FullyConnectedNet(nn.Module):
    def __init__(self):
        super(FullyConnectedNet, self).__init__()
        # 展平后的特征数量: 150x150x3
        self.input_size = 150 * 150 * 3
        # 定义全连接层
        self.fc_layers = nn.Sequential(
            nn.Linear(self.input_size, 2048),
            nn.ReLU(),
            nn.Dropout(0.5),  # 添加Dropout防止过拟合
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2),
            # nn.Softmax(dim=1)
        )
    def forward(self, x):
        # 将输入图像展平
        x = x.view(-1, self.input_size)
        # 通过全连接层
        x = self.fc_layers(x)
        return x