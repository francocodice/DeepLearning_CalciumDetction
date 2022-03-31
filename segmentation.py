import torch
import dataset
from transform import *
from utils import *
from model import UNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

path_plot = '/home/fiodice/project/plot_transform/img'

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


def show_samples(count, input, output):
    input_img = transforms.ToPILImage()(input).convert("RGB")
    output_map = decode_segmap(torch.argmax(output, dim=0))
    
    f = plt.figure()
    ax1 = f.add_subplot(1, 2, 2)
    ax1.title.set_text('Output')
    plt.imshow(output_map)

    ax3 = f.add_subplot(1, 2, 1)
    ax3.title.set_text('Input')
    plt.imshow(input_img)

    plt.show(block=True)
    plt.savefig(path_plot + str(count) + '.png')



if __name__ == '__main__':
    path_train_data = '/home/fiodice/project/dataset/'
    path_labels = '/home/fiodice/project/dataset/site.db'
    path_model = '/home/fiodice/project/model/segmentation_model.pt'


    transform = transforms.Compose([ transforms.Resize(2048),
                                     transforms.CenterCrop(2000),
                                     transforms.Resize(512),
                                     transforms.ToTensor()])

    # Normalization here decrese accuracy of the model and precision 
    # of mean, std overall the dataset.

    cac_dataset = dataset.CalciumDetection(path_train_data, path_labels, transform)

    model = UNet(in_channels=1, out_channels=6, init_features=32)
    # output channel is the number of masks obtained (READ CLASS DATASET)
    model.load_state_dict(torch.load(path_model, map_location=device))
    model.eval()
    model.to(device)

    size_train = 0.80

    train_set, test_set = split_train_val(size_train, cac_dataset)

    #mean, std = mean_std(train_set)
    #print(mean, std)
    #mean, std = [0.592], [0.192]

    #test_set = normalize(test_set, mean, std)
    
    test_loader = torch.utils.data.DataLoader(test_set,
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
            if batch_idx % 3 == 0:
                show_samples(batch_idx, data[0].cpu(), output_norm[0].cpu())




