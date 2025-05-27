import torch.nn as nn

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 28 * 28, 512)  # 注意这里的输入维度
        self.fc2 = nn.Linear(512, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))  # 112x112
        x = self.pool(self.relu(self.conv2(x)))  # 56x56
        x = self.pool(self.relu(self.conv3(x)))  # 28x28
        x = x.view(-1, 64 * 28 * 28)  # 展平
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


