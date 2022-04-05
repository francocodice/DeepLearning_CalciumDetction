import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from metrics import dice_coef_multilabel

def decode_segmap(seg_mask, nc=4):
    label_colors = np.array([(128, 0, 128),     # 0 (fuchsia) = heart
                             #(128, 0, 0),       # 1 (red)= right clavicle
                             (0, 128, 0),       # 2 (green)= right lung
                             #(128, 128, 0),     # 3 (yellow) = left clavicle
                             (0, 0, 128),       # 4 (blue) = left lung
                             (0, 0, 0)])        # 5 (black) -> background

    r = np.zeros_like(seg_mask).astype(np.uint8)
    g = np.zeros_like(seg_mask).astype(np.uint8)
    b = np.zeros_like(seg_mask).astype(np.uint8)

    for l in range(0, nc):
        idx = seg_mask == l
        r[idx] = label_colors[l, 0]
        g[idx] = label_colors[l, 1]
        b[idx] = label_colors[l, 2]

    rgb = np.stack([r, g, b], axis=2)
    return rgb


def show_samples(input, output, groundtruth, id):
    #input = torch.cat((input, (torch.sum(input, dim=0, keepdim=True) == 0)), 0)
    #output = torch.cat((output, (torch.sum(output, dim=0, keepdim=True) == 0)), 0)
    input_img = transforms.ToPILImage()(input).convert("L")

    output_map = decode_segmap(torch.argmax(output, dim=0), nc=4)
    groundtruth_map = decode_segmap(torch.argmax(groundtruth, dim=0), nc=4)
    
    f = plt.figure()
    
    ax1 = f.add_subplot(1, 3, 1)
    ax1.title.set_text('Output')
    ax1.grid(False)
    plt.imshow(output_map)
    
    ax2 = f.add_subplot(1, 3, 2)
    ax2.title.set_text('Groundtruth')
    ax2.grid(False)
    plt.imshow(groundtruth_map)
    
    ax3 = f.add_subplot(1, 3, 3)
    ax3.title.set_text('Input')
    ax3.grid(False)
    plt.imshow(input_img, cmap='gray')
    
    plt.tight_layout()
    plt.show()
    plt.savefig('/home/fiodice/project/plot_transform/' + 'seg_sample' + str(id) + '.png')



