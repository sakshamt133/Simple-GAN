import torch.nn as nn
from block import DownSample


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            DownSample(3, 8, (3, 3), (1, 1)),
            DownSample(8, 16, (3, 3), (2, 2)),
            DownSample(16, 32, (3, 3), (2, 2)),
            DownSample(32, 64, (3, 3), (2, 2), (2, 2)),
            nn.Flatten()
        )
        self.disc2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, 128),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.disc(x)
        out = out.view(out.shape[0], -1)
        return self.disc2(out)
