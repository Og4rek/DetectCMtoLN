from torch.utils import data
from torchvision import datasets
from torchvision import transforms


class Dataset:
    def __init__(self, root, batch_size):
        self.root = root
        self.batch_size = batch_size

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Grayscale(),
            transforms.RandomHorizontalFlip(p=0.5),  # Horizontal flip with probability 1.0 (always)
            transforms.RandomVerticalFlip(p=0.5),  # Vertical flip with probability 1.0 (always)
            transforms.RandomRotation(degrees=90),  # Rotation range of 90 degrees
            transforms.RandomResizedCrop(size=(96, 96), scale=(0.8, 1.1), ratio=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.03),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
            # transforms.Normalize(mean=[0.5],
            #                      std=[0.5])
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
        self.valid_data = datasets.PCAM(self.root, split='val', transform=self.transform)
        self.test_data = datasets.PCAM(self.root, split='test', transform=self.transform)

    def create_dataloaders(self):
        self.train_dataloader = data.DataLoader(self.valid_data, batch_size=self.batch_size)
        self.valid_dataloader = data.DataLoader(self.valid_data, batch_size=self.batch_size)
        self.test_dataloader = data.DataLoader(self.test_data, batch_size=self.batch_size)

