
import cv2
from skimage.morphology import ( dilation, closing, opening)
import numpy as np
from skimage.draw import disk
import torch
from utils import show

PATH_PLOT = '/home/fiodice/project/plot_morpho_elab/'

def multi_dil(im, num, elem):
    for _ in range(num):
        im = dilation(im, elem)
    return im


## Opening -> Dilatation -> Closing --> Multi Dilatation
def dilate_heart_mask(mask, cac_id, visualize=False):
    new_mask = np.zeros(mask.shape).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(12,12))

    n_regions, _ = cv2.connectedComponents(mask.cpu().numpy().astype(np.uint8), 8)

    # if net not inferred heart shadow heart, put a circle a the center of image
    if n_regions == 1:
        new_mask[disk((256, 256), 60)] = 1
        return torch.tensor(new_mask)
    else:
        # small erosion for remove noise 
        open_mask = opening(mask.cpu(), kernel)
        open_mask = multi_dil(open_mask, 1, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(12,12)))
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(open_mask.astype(np.uint8) , 8 , cv2.CV_32S)
        # if after opening there is no more then 2 element only noise mask
        if num_labels == 1:
            new_mask[disk((256, 256), 60)] = 1
            return torch.tensor(new_mask)
        else:
            max_label, _ = max([(i, stats[i, cv2.CC_STAT_AREA]) for i in range(1, num_labels)], key=lambda x: x[1])
            new_mask[labels == max_label] = 1
            new_mask = closing(new_mask, np.ones((15, 15)))
            new_mask = multi_dil(new_mask, 3, np.ones((12, 12)))
        
    if visualize:
        show([mask, new_mask], str(cac_id) + '_heart', path=PATH_PLOT)

    return torch.tensor(new_mask)

## Opening -> Closing 
def opening_lung(mask, cac_id, orientation='right', visualize=False):
    new_mask = np.zeros(mask.shape).astype(np.uint8)
    kernel = np.ones((22, 22))

    open_mask = opening(mask.cpu(), kernel)
    n_regions, labels, stats, _ = cv2.connectedComponentsWithStats(open_mask.astype(np.uint8) , 4 , cv2.CV_32S)

    if n_regions == 1:
        new_mask = np.copy(mask.cpu())
    else:
        max_label, _ = max([(i, stats[i, cv2.CC_STAT_AREA]) for i in range(1, n_regions)], key=lambda x: x[1])
        new_mask[labels == max_label] = 1
        new_mask = closing(new_mask, kernel)

    if visualize:
        show([mask, new_mask], str(cac_id) + '_lung_'  + str(orientation), path=PATH_PLOT)


    return torch.tensor(new_mask)
