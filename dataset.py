import torch
from PIL import Image
import glob
import os
import gc

import torchvision
import pydicom
import numpy as np
import sqlite3
from utils import convert
from pydicom.pixel_data_handlers.util import  apply_windowing, apply_modality_lut
from utils import *
from skimage import exposure

PATH_PLOT = '/home/fiodice/project/plot_training/'


def get_patient_id(dimg):
    #if dimg.PatientID == 'CAC_097':
    #    return 'CAC_097'
    if dimg.PatientID == 'CAC_1877':
        return dimg.PatientName
    else:
        return dimg.PatientID


class CalciumDetection(torch.utils.data.Dataset):
    def __init__(self, data_dir, labels_path, transform=None, require_path_file=False):
        self.root = data_dir
        self.elem = glob.glob(self.root + '*' + '/rx/')

        conn = sqlite3.connect(labels_path)
        conn.row_factory = sqlite3.Row  
        cursor = conn.cursor()

        self.labels = [dict(row) for row in cursor.execute('SELECT * FROM patient').fetchall()]
        self.transform = transform
        self.require_path_file = require_path_file


    def __len__(self):
        return len(self.elem)

    def __getitem__(self, idx):
        path = self.elem[idx] + os.listdir(self.elem[idx])[0]
        dimg = pydicom.dcmread(path, force=True)
        img16 = apply_windowing(dimg.pixel_array, dimg)   
        img_eq = exposure.equalize_hist(img16)
        img8 = convert(img_eq, 0, 255, np.uint8)
        img_array = ~img8 if dimg.PhotometricInterpretation == 'MONOCHROME1' else img8
        img = Image.fromarray(img_array)
        
        cac_score = [label for label in self.labels if label['id'] == get_patient_id(dimg)][0]['cac_score']
        label = 0 if int(cac_score) in range(0, 11) else 1

        if self.transform is not None:
            img = self.transform(img=img)
        else:
            img = torchvision.transforms.ToTensor()(img)

        if self.require_path_file:
            return img, path, label
        else:
            return img, label



class CalciumDetectionRegression(torch.utils.data.Dataset):
    def __init__(self, data_dir, labels_path, transform=None):
        self.root = data_dir
        self.elem = glob.glob(self.root + '*' + '/rx/')

        conn = sqlite3.connect(labels_path)
        conn.row_factory = sqlite3.Row  
        cursor = conn.cursor()

        self.labels = [dict(row) for row in cursor.execute('SELECT * FROM patient').fetchall()]
        self.cac_scores = np.array([patient['cac_score'] for patient in self.labels])
        self.transform = transform


    def __len__(self):
        return len(self.elem)

    def __getitem__(self, idx):
        # Process img                
        path = self.elem[idx] + os.listdir(self.elem[idx])[0]
        dimg = pydicom.dcmread(path, force=True)
        img16 = apply_windowing(dimg.pixel_array, dimg)
        img_eq = exposure.equalize_hist(img16)
        img8 = convert(img_eq, 0, 255, np.uint8)
        img_array = ~img8 if dimg.PhotometricInterpretation == 'MONOCHROME1' else img8
        img = Image.fromarray(img_array)

        # Process label                
        cac_score = [label for label in self.labels if label['id'] == get_patient_id(dimg)][0]['cac_score']
        #cac_clip = np.clip([cac_score],a_min=0, a_max=2000)
        #log_cac_score = np.log(cac_clip + 1)[0] 
        #cac_log = np.log((np.clip([cac_score],a_min=0, a_max=2000) + 1))
        #cac_norm = norm_log(cac_log)[0]

        if self.transform is not None:
            img = self.transform(img=img)
        else:
            img = torchvision.transforms.ToTensor()(img)

        return img.float(), cac_score 



if __name__ == '__main__':

    gc.collect()

    torch.cuda.empty_cache()

    path_data = '/home/fiodice/project/dataset/'
    #path_labels = '/home/fiodice/project/dataset/site.db'
    path_labels =  '/home/fiodice/project/labels/labels_new.db'

    mean, std = [0.5024], [0.2898]
    train_t, test_t = get_transforms(img_size=1248, crop=1024, mean = mean, std = std)

    dataset = CalciumDetection(path_data, path_labels, transform=train_t)

    data_loader = torch.utils.data.DataLoader(dataset,
                            batch_size = 1,
                            shuffle = False,
                            num_workers = 0)

    loaders = [data_loader]
    scores = []

    for loader in loaders:
        for batch_idx, (data, labels) in enumerate(loader):
            scores.append(labels.numpy()[0])

    plt.figure()
    plt.rcParams.update({'figure.figsize':(7,5), 'figure.dpi':100})
    plt.hist(scores, bins=5)
    plt.gca().set(title='Frequency Histogram', ylabel='Calcium score')
    plt.savefig(PATH_PLOT + 'hist.png')
    plt.close()
