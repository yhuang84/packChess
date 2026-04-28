from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from board_utils import NUM_POLICY_CLASSES

class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.se = SEBlock(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)
        out = out + residual
        out = F.relu(out, inplace=True)
        return out

class ChessNet(nn.Module):
    # Scaled up for V5 Large: 15 blocks and 256 channels
    def __init__(self, num_res_blocks: int = 15, channels: int = 256):
        super().__init__()
        self.initial_conv = nn.Conv2d(14, channels, kernel_size=3, padding=1, bias=False)
        self.initial_bn = nn.BatchNorm2d(channels)
        self.res_blocks = nn.ModuleList([ResBlock(channels) for _ in range(num_res_blocks)])
        self.policy_conv = nn.Conv2d(channels, 73, kernel_size=1, bias=True)

        self.value_conv = nn.Conv2d(channels, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(8 * 8, 128)
        self.value_fc2 = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor):
        x = self.initial_conv(x)
        x = self.initial_bn(x)
        x = F.relu(x, inplace=True)
        for block in self.res_blocks:
            x = block(x)
        policy_logits = self.policy_conv(x)
        policy_logits = policy_logits.view(policy_logits.size(0), NUM_POLICY_CLASSES)
        value = self.value_conv(x)
        value = self.value_bn(value)
        value = F.relu(value, inplace=True)
        value = value.view(value.size(0), 8 * 8)
        value = F.relu(self.value_fc1(value), inplace=True)
        value = torch.tanh(self.value_fc2(value))
        return policy_logits, value