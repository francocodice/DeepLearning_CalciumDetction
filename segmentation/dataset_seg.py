
import torch
from PIL import Image
import glob
import os
import torchvision
import numpy as np
from pydicom.pixel_data_handlers.util import apply_modality_lut, apply_windowing
from torchvision import transforms

def convert(img, target_type_min, target_type_max, target_type):
    imin = img.min()
    imax = img.max()

    #print(f'Convert method min {imin} max {imax}')
    
    a = (target_type_max - target_type_min) / (imax - imin)
    b = target_type_max - a * imax
    new_img = (a * img + b).astype(target_type)

    return new_img

class HeartSegmentation(torch.utils.data.Dataset):
    def __init__(self, img_dir, mask_dir, transform=None, crossentropy_prepare=False):
        self.root = img_dir
        self.mask = mask_dir
        self.mask_order = ['heart', 'left clavicle', 'left lung', 'right clavicle', 'right lung']
        self.transform = transform  ##to be implemented
        self.elem = glob.glob(self.root + '*')
        self.crossentropy_prepare = crossentropy_prepare


    def __len__(self):
        return len(self.elem)

    def __getitem__(self, idx):
        path = self.elem[idx]
        #print(path)
        data_reshaped = np.fromfile(path, dtype=">u2")
        #print("Size of this file is ", data_reshaped.shape)

        if data_reshaped.shape[0] < 4194304:
            temp = np.zeros(4194304)
            temp[:data_reshaped.shape[0]] = data_reshaped
            data_reshaped = temp

        data_reshaped = data_reshaped.reshape(2048, 2048)
        img8 = convert(data_reshaped, 0, 255, np.uint8)
        img = Image.fromarray(img8)

        name_for_masks = path.split('/')[-1].split('.')[0] + '.png'
        these_masks = [Image.open(self.mask + fold + '/' + name_for_masks).convert('L') for fold in self.mask_order]

        # image to tensor
        if self.transform is not None:
            this_image = self.transform(img=img)
        else:
            this_image = torchvision.transforms.ToTensor()(img)
        #this_image = torchvision.transforms.Normalize((0.5,), (1.0,))(this_image)  # fix these values

        # piling tensors
        these_masks_tensorized = [torchvision.transforms.ToTensor()(this_mask).squeeze(dim=0) for this_mask in
                                  these_masks]
        masks = torch.stack(these_masks_tensorized)

        if self.crossentropy_prepare:
            masks = torch.cat((masks, (torch.sum(masks, dim=0, keepdim=True) == 0)), 0)
            masks = torch.argmax(masks, dim=0, keepdim=True).type(torch.long)

        return this_image, masks


if __name__ == '__main__':
    path_data = '/home/fiodice/project/dataset_seg/data/'
    path_labels = '/home/fiodice/project/dataset_seg/mask/'

    transform = transforms.Compose([ transforms.Resize((512,512)),
                                     transforms.ToTensor()])

    whole_dataset = HeartSegmentation(path_data, path_labels, transform=transform)
    
    train_loader = torch.utils.data.DataLoader(whole_dataset,
                            batch_size = 1,
                            shuffle = False,
                            num_workers = 0)


    # TEST THE MODEL
    n_batch = len(train_loader)
    for batch_idx, (data, labels) in enumerate(train_loader):
        print(data.shape, labels.shape)