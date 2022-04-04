import pydicom as dcm
import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm, trange
import collections

IDX_IMG = 0
IDX_LABEL = 1

def mean_std(dataset):
    loader = torch.utils.data.DataLoader(dataset,
                                            batch_size=1,
                                            shuffle=False,
                                            num_workers=0)
    nimages, mean, std  = 0, 0., 0.
    print('='*15, f'Calc mean and std','='*15)
    for batch, _ in tqdm(loader):
        # Rearrange batch to be the shape of [B, C, W * H]
        batch = batch.view(batch.size(0), batch.size(1), -1)
        # Update total number of images
        nimages += batch.size(0)
        # Compute mean and std here
        mean += batch.mean(2).sum(0) 
        std += batch.std(2).sum(0)
    mean /= nimages
    std /= nimages
    return mean, std


def normalize(dataset, mean, std):
    dataset_norm = []
    print('='*15, f'Normalizing dataset','='*15)
    for j in trange(len(dataset)):
        label = dataset[j][1]
        img = dataset[j][0]
        img_norm = (img - mean[0]) / (std[0])
        dataset_norm.append((img_norm,label))
    return dataset_norm


def split_train_val(size_train, dataset):
    train_size = int(size_train * len(dataset))
    test_size = len(dataset) - train_size
    return torch.utils.data.random_split(dataset, [train_size,test_size])


def show_distribution(dataloader, set):
    batch_labels = [label.tolist() for _, label in dataloader]
    label_flat_list = [item for sublist in batch_labels for item in sublist]
    count_labels = collections.OrderedDict(sorted(collections.Counter(label_flat_list).items()))
    print(f'For {set} Labels {count_labels}')


def convert(img, target_type_min, target_type_max, target_type):
    imin = img.min()
    imax = img.max()

    #print(f'Convert method min {imin} max {imax}')
    
    a = (target_type_max - target_type_min) / (imax - imin)
    b = target_type_max - a * imax
    new_img = (a * img + b).astype(target_type)

    return new_img


def windowing(img, window_center, window_width, intercept=0, slope=1):
    # To HU
    img = (img*slope + intercept)
    
    img_min = window_center - window_width//2 #minimum HU level
    img_max = window_center + window_width//2 #maximum HU level
    
    img[img < img_min] = img_min 
    img[img > img_max] = img_max 
    
    #print(f'Img_w MIN {img_min} and MAX {img_max}')

    return img

  
def windowing_param(data):
    w_param = [data[('0028','1050')].value, #window center
                data[('0028','1051')].value] #window width
                    
    return to_int(w_param[0]),to_int(w_param[1])

def to_int(x):
    if type(x) == dcm.multival.MultiValue: return int(x[0])
    else: return int(x)



# No more used

def increase_dataset(training_set, factor = 6):
    train_transform = train_transforms()
    tensor_transforms = tensor_transform()
    augmented_dataset = []

    print('='*15, f'Augmentation','='*15)

    for j in trange((len(training_set))):
        # make a copy of each img without transform
        label = training_set[j][IDX_LABEL]
        base_transform = tensor_transforms(training_set[j][IDX_IMG])
        # to lose less information copy before cast to tensor
        augmented_dataset.append((base_transform,label))

        # if label 1 or 2 make factor copy of each img
        if label == 1  or label == 2:

            for k in range(factor):
                transformed_image =  (training_set[j][IDX_IMG])
                augmented_dataset.append((transformed_image,label))

                #if k == 3: plot_data(path_plot, base_transform, transformed_image, j)
        
        # if label 1 or 2 make one copy of each img
        elif label == 0 or label == 3:

            transformed_image = train_transform(training_set[j][IDX_IMG])
            augmented_dataset.append((transformed_image,label))

    return augmented_dataset