
import torch
from PIL import Image
import glob
from pydicom.pixel_data_handlers.util import apply_modality_lut, apply_windowing
from utils import *
from PIL import Image
import glob
import torchvision
import os
import pydicom
import numpy as np
import sqlite3


class CalciumDetectionSegmentationPNG(torch.utils.data.Dataset):
    def __init__(self, data_dir, labels_path, transform=None):
        self.root = data_dir
        self.elem = glob.glob(self.root+'*')

        conn = sqlite3.connect(labels_path)
        conn.row_factory = sqlite3.Row  
        cursor = conn.cursor()

        self.labels = [dict(row) for row in cursor.execute('SELECT * FROM patient').fetchall()]
        self.transform = transform

    def __len__(self):
        return len(self.elem)

    def __getitem__(self, idx):
        this_path = self.elem[idx] 
        this_image = Image.open(this_path).convert('L')
        patientID = this_path.split('/')[-1].split('.')[0]

        #this_image = torchvision.transforms.Normalize((0.5377,), (0.2436, ))(this_image) 
        cac_score = [label for label in self.labels if label['id'] == patientID][0]['cac_score']
        label = 0 if int(cac_score) in range(0, 11) else 1

        if self.transform is not None:
            this_image = self.transform(this_image) 
        else: 
            this_image = torchvision.transforms.ToTensor()(this_image)
        

        return this_image, label


if __name__ == '__main__':
    path_train_data = '/home/fiodice/project/data_resize_512/'
    path_labels = '/home/fiodice/project/dataset/site.db'

    whole_dataset = CalciumDetectionSegmentation(path_train_data, path_labels)
    
    train_loader = torch.utils.data.DataLoader(whole_dataset,
                            batch_size = 1,
                            shuffle = False,
                            num_workers = 0)


    # TEST THE MODEL
    n_batch = len(train_loader)
    for batch_idx, (data, labels) in enumerate(train_loader):
        print()
        #print(data, labels)