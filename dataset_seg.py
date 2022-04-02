
import torch
from PIL import Image
import glob
import numpy as np
from utils import convert
from pydicom.pixel_data_handlers.util import apply_modality_lut, apply_windowing
from utils import *
from PIL import Image
import glob
import torchvision


class CalciumDetectionSegmentation(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):
        self.root = data_dir
        self.elem = glob.glob(self.root+'*')
        self.transform = transform

    def __len__(self):
        return len(self.elem)

    def __getitem__(self, idx):
        this_path = self.elem[idx] 
        this_image = Image.open(this_path).convert('L')

        this_image = torchvision.transforms.ToTensor()(this_image)
        #this_image = torchvision.transforms.Normalize((0.5377,), (0.2436, ))(this_image) 

        return this_image, 1


# Cal mean std for normalize seg dataset #

if __name__ == '__main__':
    img_dir = '/home/fiodice/project/data_resize_512/'
    whole_dataset = CalciumDetectionSegmentation(img_dir)

    train_loader = torch.utils.data.DataLoader(whole_dataset,
                            batch_size = 1,
                            shuffle = False,
                            num_workers = 0)
    
    mean, std = mean_std(whole_dataset)
    print(mean, std)