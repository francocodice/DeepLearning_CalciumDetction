from sys import stderr
from this import d
from matplotlib.pyplot import box
import torch
from utils import *
from skimage.draw import disk
from model import UNet
from dataset import CalciumDetection
from transform import *
import torchvision
from torchvision.utils import draw_segmentation_masks
import torchvision.transforms.functional as F
import cv2
from torchvision.ops import *
from torchvision.utils import draw_bounding_boxes
from skimage.morphology import ( dilation, closing, opening)
import torchvision.transforms as T

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
path_plot_bounding_box = '/home/fiodice/project/elab/'

SIZE_IMAGE = 512


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


def multi_dil(im, num, elem):
    for i in range(num):
        im = dilation(im, elem)
    return im


def dilate_heart_mask(mask, batch_idx):
    new_mask = np.zeros(mask.shape).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
    plots = [mask]

    n, _, _, _ = cv2.connectedComponentsWithStats(mask.cpu().numpy().astype(np.uint8), 8)
    # 0 for the bck , 1 for heart 

    #print(f'For batch id {batch_idx} before processing nb comp {n}')

    if n == 1:
        new_mask[disk((256, 256), 60)] = 1
        return torch.tensor(new_mask)
    else:
        # small erosion for remove noise 
        open_mask = opening(mask.cpu(), kernel)
        plots.append(open_mask)
        #show(samples, 'segmap', path=path_plot)

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
    #show(plots, str(batch_idx) + '_heart', path=path_plot_bounding_box)

    return torch.tensor(new_mask)


def opening_lung(mask, lung_d):
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

    #show(samples, 'segmap', path=path_plot)
    return torch.tensor(new_mask)


def rgb_img(tensor):
    img_rgb = cv2.cvtColor(tensor.permute(1,2,0).cpu().detach().numpy() ,cv2.COLOR_GRAY2RGB)
    img_rgb = torchvision.transforms.ToTensor()(img_rgb)
    return F.convert_image_dtype(img_rgb, dtype=torch.uint8)


def build_box(boxes):
    x_min = min(boxes[0][0], boxes[1][0], boxes[2][0])
    y_min = min(boxes[0][1], boxes[1][1], boxes[2][1])
    x_max = max(boxes[0][2], boxes[1][2], boxes[2][2])
    y_max = max(boxes[0][3], boxes[1][3], boxes[2][3])
    return torch.tensor([[x_min, y_min, x_max, y_max]])


def bounding_box(batch_idx, img, masks):
    # 5 masks : 0 -> heart, 2 -> left lung, 4 -> right lungh
    obj_ids = torch.tensor([0, 2, 4], device=device)
    # split the color-encoded mask into a set of boolean masks.
    # Note that this snippet would work as well if the masks were float values instead of ints.
    masks = torch.argmax(masks, dim=0, keepdim=True).type(torch.long)
    masks = masks == obj_ids[:, None, None]

    # 0 -> heart, 1 -> left lung, 2 -> right lungh
    mask_heart = dilate_heart_mask(masks[0].int(), batch_idx)
    mask_left_lung = opening_lung(masks[1].int(),'left')
    mask_right_lung = opening_lung(masks[2].int(),'right')

    masks = torch.stack([ mask_heart, mask_left_lung, mask_right_lung])
    img_rgb = rgb_img(img)

    plot_result = []
    #for mask in masks:
    #    plot_result.append(draw_segmentation_masks(img_rgb, mask.bool(), alpha=0.8, colors="green"))

    #masks_flat = torch.argmax(masks, dim=0, keepdim=True)
    #single_boxe = masks_to_boxes(masks_flat)

    boxes = masks_to_boxes(masks)
    #heart_area = box_area(boxes[0])
    # boxes -> list of ( xmin , xmax , ymin , ymax )
    drawn_boxes = draw_bounding_boxes(img_rgb, boxes, colors="red")

    # in order to visualize all boxex
    plot_result.append(drawn_boxes)

    boxes_custom = crop_boxes(boxes)

    plot_result.append(draw_bounding_boxes(img_rgb, boxes_custom, colors="blue"))

    shadow_heart_box = build_box(boxes_custom)

    plot_result.append(draw_bounding_boxes(img_rgb, shadow_heart_box, colors="yellow"))

    box_tuple = (shadow_heart_box[0][0].item(), 
                shadow_heart_box[0][1].item(), 
                shadow_heart_box[0][2].item(),
                shadow_heart_box[0][3].item())
            
    #print(box_tuple)
    im = transforms.ToPILImage()(img).convert("L")
    im_cropped = im.crop(box_tuple)

    plot_result.append(np.array(im_cropped))

    show(plot_result, str(batch_idx) + '_boundig_box', path=path_plot_bounding_box)


def crop_boxes(boxes):
    # boxes = [ ( xmin , xmax , ymin , ymax ) ]
    # boxes[0] -> heart, boxes[1] -> left lung, boxes[2] -> right lungh
    id_left_lung = 2
    id_right_lung = 1 

    perc_width_lung_sx = (boxes[id_left_lung][2] - boxes[id_left_lung][0])/100
    perc_height_lung_sx = (boxes[id_left_lung][3] - boxes[id_left_lung][1])/100
    #print(f'For left lungh W :{perc_width_lung_sx * 100}, H {perc_height_lung_sx * 100}')

    boxes[id_left_lung][0] = boxes[id_left_lung][0] + (perc_width_lung_sx * 35) # -> OK
    boxes[id_left_lung][1] = boxes[id_left_lung][1] + (perc_height_lung_sx * 10) # -> OK
    #boxes[id_left_lung][3] = boxes[id_left_lung][3] - (perc_height_lung_sx * 10) # -> OK

    perc_width_lung_dx = (boxes[id_right_lung][2] - boxes[id_right_lung][0])/100
    perc_height_lung_dx = (boxes[id_right_lung][3] - boxes[id_right_lung][1])/100
    #print(f'For right lungh W :{perc_width_lung_dx * 100}, H {perc_height_lung_dx * 100}')

    boxes[id_right_lung][2] = boxes[id_right_lung][0] + (perc_width_lung_dx * 65) # -> OK
    boxes[id_right_lung][1] = boxes[id_right_lung][1] + (perc_height_lung_dx * 10) # -> OK
    #boxes[id_right_lung][3] = boxes[id_right_lung][3] - (perc_height_lung_dx * 10) # -> OK

    return boxes


def erose_lung_mask(boundig_box, orientation):
    if orientation == 'right':
        pass
    elif orientation == 'left':
        pass


## Test bounding box inferred from mask

if __name__ == '__main__':
    path_train_data = '/home/fiodice/project/dataset/'
    path_labels = '/home/fiodice/project/dataset/site.db'
    #path_plot = '/home/fiodice/project/plot_transform/sample'
    path_model = '/home/fiodice/project/model/segmentation_model.pt'

    transform = transforms.Compose([ transforms.Resize((SIZE_IMAGE,SIZE_IMAGE)),
                                     transforms.ToTensor()])


    cac_dataset = CalciumDetection(path_train_data, path_labels, transform)

    model = UNet(in_channels=1, out_channels=6, init_features=32)
    model.load_state_dict(torch.load(path_model, map_location=device))
    model.eval()
    model.to(device)


    #mean, std = mean_std(cac_dataset)
    mean, std = [0.5719], [0.2098] 
    #print(mean, std)
    #dataset = normalize(cac_dataset, mean, std)

    
    test_loader = torch.utils.data.DataLoader(cac_dataset,
                                            batch_size=1,
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
            if batch_idx == 40:
                break

