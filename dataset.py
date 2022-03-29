
import torch
from PIL import Image
import glob
import os
import pydicom
import numpy as np
import sqlite3
from utils import convert
from pydicom.pixel_data_handlers.util import apply_modality_lut, apply_windowing
from transform import *
import torchvision.transforms as transforms

SIZE_IMAGE = 1024

class CalciumDetection(torch.utils.data.Dataset):
    def __init__(self, data_dir, labels_path, transform=None):
        self.root = data_dir
        self.elem = glob.glob(self.root + '*' + '/rx/')

        conn = sqlite3.connect(labels_path)
        conn.row_factory = sqlite3.Row  
        cursor = conn.cursor()

        self.labels = [dict(row) for row in cursor.execute('SELECT * FROM patient').fetchall()]
        self.transform = transform


    def __len__(self):
        return len(self.elem)

    def __getitem__(self, idx):
        path = self.elem[idx] + os.listdir(self.elem[idx])[0]
        #print(path)
        dimg = pydicom.dcmread(path, force=True)
        img16 = apply_windowing(dimg.pixel_array, dimg)   
        #hu = apply_modality_lut(dimg.pixel_array, dimg)  
        #w_center, w_width = windowing_param(dimg) 
        #img16 = windowing(hu, w_center, w_width)

        img8 = convert(img16, 0, 255, np.uint8)
        img = ~img8 if dimg.PhotometricInterpretation == 'MONOCHROME1' else img8
        #img = Image.fromarray(cv2.cvtColor(img8_pstd,cv2.COLOR_GRAY2RGB))

        # Manage label                
        cac_score = [label for label in self.labels if label['id'] == dimg.PatientID][0]['cac_score']
        label = 0 if int(cac_score) in range(0, 100) else 1

        if self.transform is not None:
            img = self.transform(Image.fromarray(img))

        return img, label


def split_train_test(size_train, cac_dataset):
    valid_size = 1 - size_train
    num_train = len(cac_dataset)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(valid_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]

    # define samplers for obtaining training and validation batches
    train_dataset = torch.utils.data.Subset(cac_dataset, train_idx)
    test_dataset = torch.utils.data.Subset(cac_dataset, valid_idx)

    return train_dataset, test_dataset
