
import cv2
from skimage.morphology import ( dilation, closing, opening)
import numpy as np
from skimage.draw import disk
import torch

def multi_dil(im, num, elem):
    for _ in range(num):
        im = dilation(im, elem)
    return im


def dilate_heart_mask(mask, batch_idx, visualize=False):
    new_mask = np.zeros(mask.shape).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
    plots = [mask]

    # 0 for the bck , 1 for heart 
    n, _, _, _ = cv2.connectedComponentsWithStats(mask.cpu().numpy().astype(np.uint8), 8)

    #print(f'For batch id {batch_idx} before processing nb comp {n}')

    if n == 1:
        new_mask[disk((256, 256), 60)] = 1
        return torch.tensor(new_mask)
    else:
        # small erosion for remove noise 
        open_mask = opening(mask.cpu(), kernel)
        plots.append(open_mask)
        #show(samples, 'segmap', path=PATH_PLOT)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(open_mask.astype(np.uint8) , 8 , cv2.CV_32S)
        
        # if after opening there is no more then 2 element only noise mask
        if num_labels == 1:
            new_mask[disk((256, 256), 60)] = 1
            return torch.tensor(new_mask)
        else:
            max_label, _ = max([(i, stats[i, cv2.CC_STAT_AREA]) for i in range(1, num_labels)], key=lambda x: x[1])
            # max_label is the background without the heart shadow zone, so i take when is differente from 
            # associate labels
            new_mask[labels == max_label] = 1
            new_mask = closing(new_mask, np.ones((15, 15)))
            new_mask = multi_dil(new_mask, 6, np.ones((15, 15)))
        
        #print(f'For batch id {batch_idx} nb comp {num_labels}')

    plots.append(new_mask)
    #show(plots, str(batch_idx) + '_heart', path=PATH_PLOT)

    return torch.tensor(new_mask)


def opening_lung(mask, lung_d, visualize=False):
    new_mask = np.zeros(mask.shape).astype(np.uint8)
    kernel = np.ones((22, 22))

    plots = [mask]
    open_mask = opening(mask.cpu(), kernel)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(open_mask.astype(np.uint8) , 4 , cv2.CV_32S)
    plots.append(open_mask)

    if num_labels == 1:
        new_mask = np.copy(mask.cpu())
    else:
        max_label, _ = max([(i, stats[i, cv2.CC_STAT_AREA]) for i in range(1, num_labels)], key=lambda x: x[1])
        new_mask[labels == max_label] = 1
        new_mask = closing(new_mask, kernel)


    plots.append(new_mask)

    #show(plots, str(batch_idx) + '_lung_mask' + lung_d, path=path_plot_bounding_box)

    return torch.tensor(new_mask)
