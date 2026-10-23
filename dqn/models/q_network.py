"""
Dueling CNN Q-Network
Input : stacked grayscale frames (4, 84, 84)
Output: Q-values cho mỗi action
"""

import torch
import torch.nn as nn


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_size):
        super().__init__()

        in_channels = state_dim[0]  # 4 stacked frames

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),  # → (32,20,20)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),           # → (64,9,9)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),           # → (64,7,7)
            nn.ReLU(),
        )

        # Tính flat size
        dummy = torch.zeros(1, *state_dim)
        flat  = self.cnn(dummy).flatten(1).shape[1]

        # Dueling streams
        self.value = nn.Sequential(
            nn.Linear(flat, 512), nn.ReLU(),
            nn.Linear(512, 1)
        )
        self.advantage = nn.Sequential(
            nn.Linear(flat, 512), nn.ReLU(),
            nn.Linear(512, action_size)
        )

    def forward(self, x):
        feat = self.cnn(x).flatten(1)
        val  = self.value(feat)
        adv  = self.advantage(feat)
        return val + adv - adv.mean(dim=1, keepdim=True)
