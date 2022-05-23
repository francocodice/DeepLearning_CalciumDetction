import torch
import collections
import PIL
import random 
import os

import pydicom as dcm
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
import seaborn as sns
import pandas as pd 

from tqdm import tqdm, trange
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from pydicom.pixel_data_handlers.util import  apply_windowing
from PIL import Image
from skimage import exposure


IDX_IMG = 0
IDX_LABEL = 1

## dataset utils

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

## seed utils


## dicom utils

def dicom_img(path):
    dimg = dcm.dcmread(path, force=True)
    img16 = apply_windowing(dimg.pixel_array, dimg)   
    img_eq = exposure.equalize_hist(img16)
    img8 = convert(img_eq, 0, 255, np.uint8)
    img_array = ~img8 if dimg.PhotometricInterpretation == 'MONOCHROME1' else img8
    return Image.fromarray(img_array), dimg.PatientID


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


## show utils


def show_distribution(dataloader, set, path_plot):
    batch_labels = [label.tolist() for _, label in dataloader]
    label_flat_list = [item for sublist in batch_labels for item in sublist]
    count_labels = collections.OrderedDict(sorted(collections.Counter(label_flat_list).items()))
    
    val_samplesize = pd.DataFrame.from_dict(
    {'[0:100]': [count_labels[0]], 
     '> 100': count_labels[1],
    })

    sns.barplot(data=val_samplesize)
    plt.savefig(path_plot + str(set) + '.png')
    plt.close()
    print(f'For {set} Labels {count_labels}')


def show_distribution_fold(dataloader, set, fold, path_plot):
    batch_labels = [label.tolist() for _, label in dataloader]
    label_flat_list = [item for sublist in batch_labels for item in sublist]
    count_labels = collections.OrderedDict(sorted(collections.Counter(label_flat_list).items()))
    
    val_samplesize = pd.DataFrame.from_dict(
    {'[0:100]': [count_labels[0]], 
     '> 100': count_labels[1],
    })

    sns.barplot(data=val_samplesize)
    plt.savefig(path_plot + str(set) + '_fold' + str(fold) + '.png')
    plt.close()
    print(f'For {set} Labels {count_labels}')


def show(imgs, name_file, path):
    if not isinstance(imgs, list):
        imgs = [imgs]
    _, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        if type(img) == torch.tensor:
            img = img.detach()
        if type(img) != PIL.Image.Image:
            img = F.to_pil_image(img)
        #axs[0, i].imshow(np.asarray(img))
        axs[0, i].imshow(np.asarray(img), cmap='gray')
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.tight_layout()
    plt.savefig(path + name_file + '.png')
    plt.close()


def save_metric(train, test, metric, path_plot):
    plt.figure(figsize=(16, 8))
    plt.plot(train, label='Train ' + str(metric))
    plt.plot(test, label='Test ' + str(metric))
    plt.legend()
    plt.savefig(path_plot + str(metric) + '.png')
    plt.close()
    

def save_losses(train_losses, test_losses, best_test_acc, path_plot):
    plt.figure(figsize=(16, 8))
    plt.title(f'Best accuracy : {best_test_acc:.4f}')
    plt.plot(train_losses, label='Train loss')
    plt.plot(test_losses, label='Test loss')
    plt.legend()
    plt.savefig(path_plot  + 'losses.png')
    plt.close()


def save_cm(true_labels, best_pred_labels, path_plot):
    cm = confusion_matrix(true_labels, best_pred_labels)
    ax = sns.heatmap(cm, annot=True, fmt="d")
    hm = ax.get_figure()
    hm.savefig(path_plot + 'cm.png')
    hm.clf()
    plt.close(hm)


def save_roc_curve(true_labels, max_probs, path_plot):
    fpr, tpr, _ = roc_curve(true_labels, max_probs, pos_label=1)
    roc_auc = auc(fpr, tpr)
    plt.figure(1)
    lw = 2
    plt.plot(fpr, tpr, color="darkorange", label='AUC = %0.2f' % roc_auc)
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc="lower right")
    plt.savefig(path_plot  + 'roc.png')
    plt.close()

def save_roc_curve_fold(true_labels, probs, fold, path_plot):
    fpr, tpr, _ = roc_curve(true_labels, probs, pos_label=1)
    roc_auc = auc(fpr, tpr)
    plt.figure(1)
    lw = 2
    plt.plot(fpr, tpr, color="darkorange", label='AUC = %0.2f' % roc_auc)
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc="lower right")
    plt.savefig(path_plot  + 'fold_' + str(fold) + 'roc.png')
    plt.close()