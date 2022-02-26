import torch.nn as nn
from block import Block


class DCGAN(nn.Module):
    def __init__(self, noise_dim):
        super(DCGAN, self).__init__()
        self.model = nn.Sequential(
            Block(noise_dim, 64, (3, 3), (1, 1)),
            Block(64, 32, (3, 3), (2, 2)),
            Block(32, 16, (3, 3), (2, 2), (1, 1)),
            Block(16, 8, (3, 3), (2, 2), (1, 1)),
            Block(8, 3, (4, 4), (1, 1))
        )

    def forward(self, x):
        return self.model(x)
