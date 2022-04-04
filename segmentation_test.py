import torch
from utils import *
from model import UNet
from dataset_png import CalciumDetectionSegmentationPNG

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

path_plot = '/home/fiodice/project/example_segmentation/imgnorm'

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



if __name__ == '__main__':
    path_data = '/home/fiodice/project/data_resize_512/'
    path_model = '/home/fiodice/project/model/segmentation_model.pt'

    # Read dicom file as PNG file stored in data_resize_512 e data_resize_2048
    seg_dataset = CalciumDetectionSegmentationPNG(path_data)

    model = UNet(in_channels=1, out_channels=5, init_features=32)

    model.load_state_dict(torch.load(path_model, map_location=device))
    model.eval()
    model.to(device)

    mean, std = mean_std(seg_dataset)
    #mean, std = [0.5884], [0.1927]
    #print(mean, std)
    dataset = normalize(seg_dataset, mean, std)

    
    test_loader = torch.utils.data.DataLoader(seg_dataset,
                                            batch_size=4,
                                            shuffle=False,
                                            num_workers=0)

    # TEST THE MODEL
    n_batch = len(test_loader)
    for batch_idx, (data, labels) in enumerate(test_loader):
        data, labels = data.to(device), labels.to(device)
        with torch.no_grad():
            output = model(data)
            output_norm = torch.nn.functional.softmax(output, dim=1)
            show_samples_seg(batch_idx, data[0].squeeze(0).cpu(), output_norm[0].cpu())

            if batch_idx == 50:
                break