import torchvision.transforms as transforms
import albumentations
import albumentations.pytorch
from albumentations.pytorch import ToTensorV2
import torch


class TrasformDataset(torch.utils.data.Dataset):
    """
    Given a dataset, creates a dataset which applies a mapping function
    to its items (lazily, only when an item is called).

    Note that data is not cloned/copied from the initial dataset.
    """

    def __init__(self, dataset, map_fn):
        self.dataset = dataset
        self.map = map_fn

    def __getitem__(self, index):
        if self.map:     
            img = self.dataset[index][0]
            #x = self.map(image=img)['image'] -> for albumentation
            x = self.map(image=img)

        else:     
            x = self.dataset[index][0]  # image
        y = self.dataset[index][1]   # label      
        return x, y

    def __len__(self):
        return len(self.dataset)        


def get_transforms(img_size, mean, std):
    train_transforms = albumentations.Compose([
        albumentations.Resize(img_size, img_size, always_apply=True),
        #albumentations.HorizontalFlip(p=0.3),
        albumentations.ShiftScaleRotate(shift_limit=0.1, scale_limit=1.5),
        albumentations.Normalize(mean, std),
        ToTensorV2(),
    ])

    test_transform = albumentations.Compose([
        albumentations.Resize(img_size, img_size, always_apply=True),
        albumentations.Normalize(mean, std),
        ToTensorV2(),
    ])
    
    return train_transforms, test_transform


def train_transforms():
    return transforms.Compose([
        transforms.RandomApply([
            transforms.RandomRotation(20),
            transforms.RandomAffine(0, translate=(0, 0.1), scale=(1, 1.10)) ],
            0.8),
        transforms.Resize(512),
		transforms.ToTensor()])