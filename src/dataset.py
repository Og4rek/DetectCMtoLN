from torch.utils import data
from torchvision import datasets
from torchvision.transforms import v2
import torchvision.transforms.functional as F
import random


class RandomZoom(object):
    def __init__(self, zoom_range=1.1, p=0.75):
        self.zoom_range = zoom_range
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            scale = random.uniform(1.0, self.zoom_range)
            return F.affine(img, angle=0, translate=[0, 0], scale=scale, shear=[0, 0])
        return img


class RandomAffineWithWrap(object):
    def __init__(self, max_wrap=0.2, p=0.75):
        self.max_wrap = max_wrap
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            shear_factor = random.uniform(-self.max_wrap, self.max_wrap)
            return F.affine(img, angle=0, translate=[0, 0], scale=1.0, shear=[shear_factor, shear_factor])
        return img


class Dataset:
    def __init__(self, root, batch_size):
        self.root = root
        self.batch_size = batch_size

        self.transform = v2.Compose([
            v2.ToTensor(),
            v2.RandomHorizontalFlip(p=0.5),  # Horizontal flip with probability 1.0 (always)
            v2.RandomVerticalFlip(p=0.5),  # Vertical flip with probability 1.0 (always)
            v2.RandomRotation(degrees=90),  # Rotation range of 90 degrees
            RandomZoom(zoom_range=1.1, p=0.75),
            RandomAffineWithWrap(max_wrap=0.2, p=0.75),
            v2.ColorJitter(brightness=0.03, contrast=0.03, saturation=0.03, hue=0.03),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.transform_valid = v2.Compose([
            v2.ToTensor(),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.test_data = None
        self.valid_data = None
        self.training_data = None

        self.test_dataloader = None
        self.valid_dataloader = None
        self.train_dataloader = None

        self.create_datasets()
        self.create_dataloaders()

    def create_datasets(self):
        self.training_data = datasets.PCAM(self.root, split='train', transform=self.transform)
        self.valid_data = datasets.PCAM(self.root, split='val', transform=self.transform_valid)
        self.test_data = datasets.PCAM(self.root, split='test', transform=self.transform_valid)

    def create_dataloaders(self):
        self.train_dataloader = data.DataLoader(self.training_data, batch_size=self.batch_size)
        self.valid_dataloader = data.DataLoader(self.valid_data, batch_size=self.batch_size)
        self.test_dataloader = data.DataLoader(self.test_data, batch_size=self.batch_size)