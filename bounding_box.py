from matplotlib.pyplot import box
import torch
from model import UNet
from dataset import CalciumDetection
import torchvision
from torchvision.utils import draw_segmentation_masks
import torchvision.transforms.functional as F
import cv2
from torchvision.utils import draw_bounding_boxes
import pydicom as dcm
from pydicom.pixel_data_handlers.util import  apply_windowing
from PIL import Image
from morpho_elab import *
from utils import *
from torchvision.ops import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

OUTPUT_FOLDER_TRAIN = '/home/fiodice/project/dataset_png/train/'
OUTPUT_FOLDER_TEST = '/home/fiodice/project/dataset_png/test/'

PATH_PLOT = '/home/fiodice/project/elab/'

SIZE_IMAGE = 512


def rgb_img(tensor):
    img_rgb = cv2.cvtColor(tensor.permute(1,2,0).cpu().detach().numpy() ,cv2.COLOR_GRAY2RGB)
    img_rgb = torchvision.transforms.ToTensor()(img_rgb)
    return F.convert_image_dtype(img_rgb, dtype=torch.uint8)


def get_heart_shadow_box(boxes):
    x_min = min(boxes[0][0], boxes[1][0], boxes[2][0])
    y_min = min(boxes[0][1], boxes[1][1], boxes[2][1])
    x_max = max(boxes[0][2], boxes[1][2], boxes[2][2])
    y_max = max(boxes[0][3], boxes[1][3], boxes[2][3])
    return torch.tensor([[x_min, y_min, x_max, y_max]])


def bounding_box(index, img, masks, scale_factor, visualize=False):
    img_rgb = rgb_img(img)

    # 5 masks : 0 -> heart, 2 -> left lung, 4 -> right lungh
    obj_ids = torch.tensor([0, 2, 4], device=device)
    masks = torch.argmax(masks, dim=0, keepdim=True).type(torch.long)
    masks = masks == obj_ids[:, None, None]

    mask_heart = dilate_heart_mask(masks[0].int(), batch_idx)
    mask_left_lung = opening_lung(masks[1].int(),'left')
    mask_right_lung = opening_lung(masks[2].int(),'right')
    masks = torch.stack([ mask_heart, mask_left_lung, mask_right_lung])

    boxes = masks_to_boxes(masks)
    boxes_custom = crop_boxes(boxes)
    shadow_heart_box = get_heart_shadow_box(boxes_custom)
    
    if visualize:
        plot_result = [draw_bounding_boxes(img_rgb, boxes, colors="red"),
                    draw_bounding_boxes(img_rgb, boxes_custom, colors="blue"),
                    draw_bounding_boxes(img_rgb, shadow_heart_box, colors="green")]
        show(plot_result, str(index) + '_boundig_box', path=PATH_PLOT)

    return shadow_heart_box * scale_factor


def crop_boxes(boxes):
    ID_HEART = 0 
    ID_LUNG_DX = 1 
    ID_LUNG_SX = 2
    c_boxes = boxes.detach().clone()

    perc_width_lung_sx = (boxes[ID_LUNG_SX][2] - boxes[ID_LUNG_SX][0]) / 100
    perc_height_lung_sx = (boxes[ID_LUNG_SX][3] - boxes[ID_LUNG_SX][1]) / 100
    perc_width_lung_dx = (boxes[ID_LUNG_DX][2] - boxes[ID_LUNG_DX][0]) / 100
    perc_height_lung_dx = (boxes[ID_LUNG_DX][3] - boxes[ID_LUNG_DX][1]) / 100
    #print(f'For left lungh W :{perc_width_lung_sx * 100}, H {perc_height_lung_sx * 100}')
    #print(f'For right lungh W :{perc_width_lung_dx * 100}, H {perc_height_lung_dx * 100}')

    c_boxes[ID_HEART] = boxes[ID_HEART]
    # decrease size of dx lung bounding box
    c_boxes[ID_LUNG_SX][0] = boxes[ID_LUNG_SX][0] + (perc_width_lung_sx * 35) 
    c_boxes[ID_LUNG_SX][1] = boxes[ID_LUNG_SX][1] + (perc_height_lung_sx * 10)
    c_boxes[ID_LUNG_SX][2] = boxes[ID_LUNG_SX][2]
    c_boxes[ID_LUNG_SX][3] = boxes[ID_LUNG_SX][3] - (perc_height_lung_sx * 15) 
    # decrease size of dx lung bounding box
    c_boxes[ID_LUNG_DX][0] = boxes[ID_LUNG_DX][0]
    c_boxes[ID_LUNG_DX][1] = boxes[ID_LUNG_DX][1] + (perc_height_lung_dx * 10) 
    c_boxes[ID_LUNG_DX][2] = boxes[ID_LUNG_DX][0] + (perc_width_lung_dx * 65) 
    c_boxes[ID_LUNG_DX][3] = boxes[ID_LUNG_DX][3] - (perc_height_lung_dx * 15) 

    return c_boxes


def dicom_img(path):
    dimg = dcm.dcmread(path, force=True)
    img16 = apply_windowing(dimg.pixel_array, dimg)   
    img8 = convert(img16, 0, 255, np.uint8)
    img_array = ~img8 if dimg.PhotometricInterpretation == 'MONOCHROME1' else img8
    return Image.fromarray(img_array)


if __name__ == '__main__':
    path_train_data = '/home/fiodice/project/dataset/'
    path_labels = '/home/fiodice/project/dataset/site.db'
    path_model = '/home/fiodice/project/model/segmentation_model.pt'

    transform = torchvision.transforms.Compose([ torchvision.transforms.Resize((SIZE_IMAGE,SIZE_IMAGE)),
                                                torchvision.transforms.ToTensor()])


    cac_dataset = CalciumDetection(path_train_data, path_labels, transform, require_path_file=True)

    model = UNet(in_channels=1, out_channels=6, init_features=32)
    model.load_state_dict(torch.load(path_model, map_location=device))
    model.eval()
    model.to(device)

    #mean, std = mean_std(cac_dataset)
    mean, std = [0.5719], [0.2098] 
    #print(mean, std)
    #dataset = normalize(cac_dataset, mean, std)

    # heart_box for img 512 x 512 to apply at 2048 x 2048
    scale_factor = 4
    visualize=False
    to_tensor = torchvision.transforms.ToTensor()

    data_loader = torch.utils.data.DataLoader(cac_dataset,
                                            batch_size = 1,
                                            shuffle=False,
                                            num_workers=0)

    for batch_idx, (data, path, labels) in enumerate(data_loader):
        data, labels = data.to(device), labels.to(device)
        output = model(data)

        img = data[0]
        masks = torch.nn.functional.softmax(output, dim=1)[0]
        path_data = path[0]

        shadow_heart_box = bounding_box(batch_idx, img, masks, scale_factor, visualize=True) 

        dimg = dicom_img(path_data)         
        dimg.thumbnail((2048, 2048), Image.ANTIALIAS)

        box = (shadow_heart_box[0][0].item(), 
                        shadow_heart_box[0][1].item(), 
                        shadow_heart_box[0][2].item(),
                        shadow_heart_box[0][3].item())
            
        img_cropped = dimg.crop(box)
        out_name = path_data.split('/')[-3]+'.png'

        if 'train' in path_data:
            img_cropped.save(OUTPUT_FOLDER_TRAIN + out_name)
        else:
            img_cropped.save(OUTPUT_FOLDER_TEST + out_name)

        print(f'Saved {out_name} with dim {img_cropped.size}')

        if visualize:
            plot_result = [dimg, 
                           draw_bounding_boxes(rgb_img(to_tensor(dimg)), shadow_heart_box, colors="red"),
                           img_cropped]
            show(plot_result, str(batch_idx) + '_crop', path=PATH_PLOT)

        if batch_idx == 10:
            break

