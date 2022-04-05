from sys import stderr
from this import d
import torch
from torch import tensor
from utils import *
from skimage.draw import disk
from model import UNet
from dataset import CalciumDetection
from transform import *
import torchvision
from torchvision.utils import draw_segmentation_masks
import torchvision.transforms.functional as F
import cv2
from torchvision.ops import masks_to_boxes
from torchvision.utils import draw_bounding_boxes
from skimage.morphology import (erosion, dilation, closing, opening,
                                area_closing, area_opening)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
path_plot = '/home/fiodice/project/example_segmentation/'
SIZE_IMAGE = 512


def show(imgs, msg, path):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        if type(img) == torch.tensor:
            img = img.detach()
        img = F.to_pil_image(img)
        #axs[0, i].imshow(np.asarray(img))
        axs[0, i].imshow(np.asarray(img), cmap='gray')
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.tight_layout()
    plt.savefig(path + msg + '.png')
    plt.close()


def multi_dil(im, num, elem):
    for i in range(num):
        im = dilation(im, elem)
    return im


def dilate_heart_mask(mask, batch_idx):
    new_mask = np.zeros(mask.shape).astype(np.uint8)

    samples = [mask]
    n, _, _, _ = cv2.connectedComponentsWithStats(mask.cpu().numpy().astype(np.uint8), 4)
    print(f'For batch id {batch_idx} before processing nb comp {n}')

    # 0 for the bck , 1 for heart 
    if n > 1:
        # small erosion for remove noise 
        close_mask = erosion(mask.cpu(), np.ones((4, 4)))
        # big dilation to be sure of take shadow heart
        dilated_mask = multi_dil(close_mask, 2, np.ones((8, 8)))

        samples.append(dilated_mask)
        #show(samples, 'segmap', path=path_plot)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(dilated_mask.astype(np.uint8) , 4 , cv2.CV_32S)
        
        if num_labels == 1:
            new_mask[disk((256, 256), 40)] = 1
        else:
            max_label, _ = max([(i, stats[i, cv2.CC_STAT_AREA]) for i in range(1, num_labels)], key=lambda x: x[1])
                # max_label is the background without the heart shadow zone, so i take when is differente from 
                # associate labels
            new_mask[labels == max_label] = 1
            new_mask = multi_dil(new_mask, 3, np.ones((10, 10)))
        

        print(f'For batch id {batch_idx} nb comp {num_labels}')
    else:
        # if the code is here the nn has not inferred the heart
        # put a circle at the center of image
        new_mask[disk((256, 256), 40)] = 1


    samples.append(new_mask)

    show(samples, 'heart_mask_' + str(batch_idx), path=path_plot)

    #cv2.imshow("Biggest component", img2)

    return torch.tensor(new_mask)


def close_lung(mask, lung_d):
    samples = [mask]

    # to remove big noise -> big structered element
    struc_element = np.ones((20, 20))
    close_mask = erosion(mask.cpu(), struc_element)
    close_mask = dilation(close_mask, struc_element)

    samples.append(close_mask)
    show(samples, 'lung_mask_' + lung_d + str(batch_idx), path=path_plot)

    #show(samples, 'segmap', path=path_plot)
    return torch.tensor(close_mask)

    


def rgb_img(tensor):
    img_rgb = cv2.cvtColor(tensor.permute(1,2,0).cpu().detach().numpy() ,cv2.COLOR_GRAY2RGB)
    img_rgb = torchvision.transforms.ToTensor()(img_rgb)
    return F.convert_image_dtype(img_rgb, dtype=torch.uint8)


def bounding_box(batch_idx, img, masks):
    # 0 -> heart, 2 -> left lung, 4 -> right lungh
    obj_ids = torch.tensor([0, 2, 4], device=device)
    # split the color-encoded mask into a set of boolean masks.
    # Note that this snippet would work as well if the masks were float values instead of ints.
    masks = torch.argmax(masks, dim=0, keepdim=True).type(torch.long)
    masks = masks == obj_ids[:, None, None]

    mask_heart = dilate_heart_mask(masks[0].int(), batch_idx)
    mask_left_lung = close_lung(masks[1].int(),'left')
    mask_right_lung = close_lung(masks[2].int(),'right')

    masks = torch.stack([mask_heart, mask_left_lung, mask_right_lung])
    img_rgb = rgb_img(img)

    drawn_masks = []
    for mask in masks:
        drawn_masks.append(draw_segmentation_masks(img_rgb, mask.bool(), alpha=0.8, colors="green"))

    boxes = masks_to_boxes(masks)
    drawn_boxes = draw_bounding_boxes(img_rgb, boxes, colors="red")

    # in order to visualize all
    drawn_masks.append(drawn_boxes)

    show(drawn_masks, 'bounding_box' + str(batch_idx), path=path_plot)



## Test bounding box inferred from mask

if __name__ == '__main__':
    path_train_data = '/home/fiodice/project/dataset/'
    path_labels = '/home/fiodice/project/dataset/site.db'
    #path_plot = '/home/fiodice/project/plot_transform/sample'
    path_model = '/home/fiodice/project/model/segmentation_model.pt'

    transform = transforms.Compose([ transforms.Resize((560,560)),
                                     transforms.CenterCrop(SIZE_IMAGE),
                                     transforms.ToTensor()])


    cac_dataset = CalciumDetection(path_train_data, path_labels, transform)

    model = UNet(in_channels=1, out_channels=6, init_features=32)
    model.load_state_dict(torch.load(path_model, map_location=device))
    model.eval()
    model.to(device)


    #mean, std = mean_std(cac_dataset)
    mean, std = [0.5884], [0.1927]
    #print(mean, std)
    #dataset = normalize(cac_dataset, mean, std)

    
    test_loader = torch.utils.data.DataLoader(cac_dataset,
                                            batch_size=4,
                                            shuffle=False,
                                            num_workers=0)

    # TEST THE MODEL
    n_batch = len(test_loader)
    for batch_idx, (data, labels) in enumerate(test_loader):
        data, labels = data.to(device), labels.to(device)
        with torch.no_grad():
            output = model(data)

            img = data[0]
            masks = torch.nn.functional.softmax(output, dim=1)[0]

            bounding_box(batch_idx, img, masks)

            if batch_idx == 50:
                break
