import torch
import torch.nn.functional as F

from PIL import Image

import os
import json
import model
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from torchvision import models
from torchvision import transforms

from captum.attr import IntegratedGradients
from captum.attr import GradientShap
from captum.attr import Occlusion
from captum.attr import NoiseTunnel
from captum.attr import visualization as viz

from model import load_densenet_mlp, HierarchicalResidual
from utils import *


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PATH_PLOT = '/home/fiodice/project/plot_training/'


def get_transforms(img_size, crop, mean, std):
    train_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        #transforms.RandomRotation(degrees=15),
        #transforms.RandomPerspective(distortion_scale=0.3, p=0.3),
        transforms.CenterCrop(crop),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.CenterCrop(crop),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    
    return train_transforms, test_transform



if __name__ == '__main__':
    path_model = '/home/fiodice/project/src/best/calcium-detection-sdg-seed-42-fold-2.pt'
    path_img = '/home/fiodice/project/dataset_split/test/CAC_001.1/rx/IM-0001-0001-0001.dcm'

    cd_model = model.test_calcium_det(path_model)

    mean, std = [0.5024], [0.2898]
    transform, _ = get_transforms(img_size=1248, crop=1024, mean = mean, std = std)

    img, _ = dicom_img(path_img)
    input = transform(img).unsqueeze(0)
    print(input.shape)

    output = cd_model(input)
    output = torch.nn.functional.softmax(output, dim=1)
    prediction_score, pred_label_idx = torch.topk(output, 1)

    pred_label_idx.squeeze_()
    print('Predicted:', pred_label_idx, '(', prediction_score.squeeze().item(), ')')

    integrated_gradients = IntegratedGradients(cd_model)
    attributions_ig = integrated_gradients.attribute(input, target=pred_label_idx, n_steps=100)

    default_cmap = LinearSegmentedColormap.from_list('custom blue', 
                                                 [(0, '#ffffff'),
                                                  (0.25, '#000000'),
                                                  (1, '#000000')], N=256)

    input_attr = np.transpose(attributions_ig.squeeze(0).cpu().detach().numpy(), (1,2,0))
    input_img = np.transpose(input.squeeze(0).cpu().detach().numpy(), (1,2,0))


    _ = viz.visualize_image_attr(input_attr,
                                input_img,
                                method='heat_map',
                                cmap=default_cmap,
                                show_colorbar=True,
                                sign='positive',
                                outlier_perc=1)