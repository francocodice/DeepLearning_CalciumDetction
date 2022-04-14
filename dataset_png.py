
import torch
from PIL import Image
import glob
import torchvision
import sqlite3
from utils import *


class CalciumDetectionPNG(torch.utils.data.Dataset):
    def __init__(self, data_dir, labels_path, transform=None):
        self.root = data_dir
        self.elem = glob.glob(self.root + '*')

        conn = sqlite3.connect(labels_path)
        conn.row_factory = sqlite3.Row  
        cursor = conn.cursor()

        self.labels = [dict(row) for row in cursor.execute('SELECT * FROM patient').fetchall()]
        self.transform = transform


    def __len__(self):
        return len(self.elem)

    def __getitem__(self, idx):
        path = self.elem[idx] 
        cac_id = path.split('/')[-1].split('.png')[-2]
        dimg = Image.open(path).convert('L')
        #dimg = Image.open(path).convert('RGB')

        cac_score = [label for label in self.labels if label['id'] == cac_id][0]['cac_score']

        #cac_score = [label for label in self.labels if label['id'] == cac_id][0]['cac_score']
        label = 0 if int(cac_score) in range(0, 11) else 1

        if self.transform is not None:
            img = self.transform(img=dimg)
        else:
            img = torchvision.transforms.ToTensor()(dimg)

        return img, label


if __name__ == '__main__':
    path_data = '/home/fiodice/project/dataset_png/test/'
    path_labels = '/home/fiodice/project/dataset/site.db'

    transform = torchvision.transforms.Compose([ torchvision.transforms.Resize((1048,1048)),
                                     torchvision.transforms.CenterCrop(1024),
                                     torchvision.transforms.ToTensor()])

    dataset = CalciumDetectionPNG(path_data, path_labels, transform=transform)
    
    loader = torch.utils.data.DataLoader(dataset,
                            batch_size = 1,
                            shuffle = False,
                            num_workers = 0)

    for batch_idx, (data, labels) in enumerate(loader):
        print(data.shape)