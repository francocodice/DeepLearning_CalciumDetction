
import imp
import torch
from PIL import Image
import glob
import os
import pydicom
import numpy as np
import sqlite3
from utils import convert
from pydicom.pixel_data_handlers.util import apply_modality_lut, apply_windowing
from utils import *


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
        label = 0 if int(cac_score) in range(0, 11) else 1

        if self.transform is not None:
            img = self.transform(Image.fromarray(img))
        else:
            img = Image.fromarray(img)

        return img, label
