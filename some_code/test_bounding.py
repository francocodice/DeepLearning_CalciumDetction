from torch import tensor
import torchvision.transforms.functional as F
import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import draw_bounding_boxes

def show(imgs, name_file, path):
    if not isinstance(imgs, list):
        imgs = [imgs]
    _, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        if type(img) == torch.tensor:
            img = img.detach()
        img = F.to_pil_image(img)
        #axs[0, i].imshow(np.asarray(img))
        axs[0, i].imshow(np.asarray(img), cmap='gray')
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.tight_layout()
    plt.savefig(path + name_file + '.png')
    plt.close()


if __name__ == '__main__':
    path_plot_bounding_box = '/home/fiodice/project/elab/'
    new_mask = torch.tensor(np.ones((1,512,512)).astype(np.uint8)) * 255

    plot = []

    boundig_box = torch.tensor([[30.,350.,200.,450.]])
    
    width = boundig_box[0][2] - boundig_box[0][0]
    height = boundig_box[0][3] - boundig_box[0][1]

    print(f'For mask W :{width}, H {height}')


    plot.append(draw_bounding_boxes(new_mask, boundig_box, colors="black"))

    boundig_box = torch.tensor([[10.,350.,100.,450.]])

    plot.append(draw_bounding_boxes(new_mask, boundig_box, colors="green"))

    show(plot, str(2) + '_bb', path=path_plot_bounding_box)

