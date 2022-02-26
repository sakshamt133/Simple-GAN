from torch.utils.data import DataLoader
from dataset import Potato
import utils

p = Potato("D:\\Datasets\\Computer Vision\\vegetable\\train\Potato", utils.transform)

train_data = DataLoader(
    p,
    utils.batch_size,
    shuffle=True
)

