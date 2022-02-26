import cv2 as cv
import os
import utils
from torch.utils.data import Dataset


class Potato(Dataset):
    def __init__(self, path, transform=None):
        self.path = path
        self.data = os.listdir(path)
        self.transform = transform

    def __getitem__(self, item):
        out = self.data[item]
        path = os.path.join(self.path, out)
        img = cv.imread(path)
        img = cv.resize(img, (utils.resize_img, utils.resize_img))
        if self.transform:
            img = self.transform(img)

        return img

    def __len__(self):
        return len(self.data)
