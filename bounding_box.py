from sys import stderr
from this import d
import torch
from utils import *
from model import UNet
from dataset import CalciumDetection
from transform import *
import torchvision
from torchvision.utils import draw_segmentation_masks
import torchvision.transforms.functional as F
import cv2
from torchvision.ops import masks_to_boxes
from torchvision.utils import draw_bounding_boxes


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
path_plot = '/home/fiodice/project/example_segmentation/xray'
SIZE_IMAGE = 512


def decode_segmap(seg_mask, nc=6):
    label_colors = np.array([(128, 0, 128),     # 0 (fucsia) = heart
                             (128, 0, 0),       # 1 (red)= right clavicle
                             (0, 128, 0),       # 2 (green)= right lung
                             (128, 128, 0),     # 3 (yellow) = left clavicle
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


def show_samples_seg(count, input, output):
    input_img = input
    output_map = decode_segmap(torch.argmax(output, dim=0))
    
    f = plt.figure()
    ax1 = f.add_subplot(1, 2, 2)
    ax1.title.set_text('Output')
    plt.imshow(output_map)

    ax3 = f.add_subplot(1, 2, 1)
    ax3.title.set_text('Input')
    plt.imshow(input_img, cmap='gray')

    plt.show(block=True)
    plt.savefig(path_plot + str(count) + '.png')
    print('plot')


def show(imgs, msg, path):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.tight_layout()
    plt.savefig(path + msg + '.png')
    plt.close()


if __name__ == '__main__':
    path_train_data = '/home/fiodice/project/dataset/'
    path_labels = '/home/fiodice/project/dataset/site.db'
    #path_plot = '/home/fiodice/project/plot_transform/sample'
    path_model = '/home/fiodice/segmentation_model_fi.pt'
    #path_model = '/home/fiodice/project/model/segmentation_model.pt'

    transform = transforms.Compose([ transforms.Resize((SIZE_IMAGE,SIZE_IMAGE)),
                                     transforms.ToTensor()])

    # Normalization here decrese accuracy of the model and precision 
    # of mean, std overall the dataset.

    cac_dataset = CalciumDetection(path_train_data, path_labels, transform)

    model = UNet(in_channels=1, out_channels=5, init_features=32)
    # output channel is the number of masks obtained (READ CLASS DATASET)
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
            masks = torch.argmax(masks, dim=0, keepdim=True).type(torch.long)

            obj_ids = torch.unique(masks)
            # to fix
            #obj_ids = torch.tensor([0, 2, 4], device=device)

            # split the color-encoded mask into a set of boolean masks.
            # Note that this snippet would work as well if the masks were float values instead of ints.
            masks = masks == obj_ids[:, None, None]

            drawn_masks = []
            img_rgb = cv2.cvtColor(img.permute(1,2,0).cpu().detach().numpy() ,cv2.COLOR_GRAY2RGB)
            img_rgb = torchvision.transforms.ToTensor()(img_rgb)
            img = F.convert_image_dtype(img_rgb, dtype=torch.uint8)

            for mask in masks:
                drawn_masks.append(draw_segmentation_masks(img, mask, alpha=0.8, colors="green"))
            

            boxes = masks_to_boxes(masks)
            drawn_boxes = draw_bounding_boxes(img, boxes, colors="red")

            drawn_masks.append(drawn_boxes)

            show(drawn_masks, 'box' + str(batch_idx), path=path_plot)

            if batch_idx == 5:
                break

