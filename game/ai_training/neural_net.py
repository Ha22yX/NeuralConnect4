import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)
        return x

class ConnectFourNet(nn.Module):
    def __init__(self, num_channels=128, num_res_blocks=10):
        super(ConnectFourNet, self).__init__()
        
        # 输入层
        self.conv_input = nn.Conv2d(1, num_channels, 3, padding=1)
        self.bn_input = nn.BatchNorm2d(num_channels)

        # 残差层
        self.res_blocks = nn.ModuleList([ResBlock(num_channels) for _ in range(num_res_blocks)])

        # 策略头
        self.conv_policy = nn.Conv2d(num_channels, 32, 1)
        self.bn_policy = nn.BatchNorm2d(32)
        self.fc_policy = nn.Linear(32 * 6 * 7, 7)  # 7列可能的动作

        # 价值头
        self.conv_value = nn.Conv2d(num_channels, 32, 1)
        self.bn_value = nn.BatchNorm2d(32)
        self.fc_value1 = nn.Linear(32 * 6 * 7, 256)
        self.fc_value2 = nn.Linear(256, 1)

    def forward(self, x):
        # 确保输入是在GPU上（如果可用）
        if next(self.parameters()).is_cuda:
            x = x.cuda()

        # 主干网络
        x = F.relu(self.bn_input(self.conv_input(x)))
        for res_block in self.res_blocks:
            x = res_block(x)

        # 策略头
        policy = F.relu(self.bn_policy(self.conv_policy(x)))
        policy = policy.view(-1, 32 * 6 * 7)
        policy = self.fc_policy(policy)
        policy = F.softmax(policy, dim=1)

        # 价值头
        value = F.relu(self.bn_value(self.conv_value(x)))
        value = value.view(-1, 32 * 6 * 7)
        value = F.relu(self.fc_value1(value))
        value = torch.tanh(self.fc_value2(value))

        return policy, value

    def predict(self, state):
        """
        对给定状态进行预测
        """
        # 将状态转换为张量
        x = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0)
        
        # 设置为评估模式
        self.eval()
        with torch.no_grad():
            policy, value = self(x)
        
        return policy.squeeze().cpu().numpy(), value.squeeze().cpu().numpy()

    @staticmethod
    def get_device():
        """
        获取可用的设备（GPU或CPU）
        """
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu') 