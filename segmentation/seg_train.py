
import torch
import numpy as np
from pydicom.pixel_data_handlers.util import apply_modality_lut, apply_windowing
from torchvision import transforms
from dataset_seg import HeartSegmentation
from sklearn.model_selection import train_test_split
from model import UNet
from tqdm import tqdm, trange
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def decode_segmap(seg_mask, nc=6):
    label_colors = np.array([(128, 0, 128),     # 0 (fuchsia) = heart
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


def show_samples(input, output, groundtruth):
    input_img = transforms.ToPILImage()(input).convert("L")
    output_map = decode_segmap(torch.argmax(output, dim=0))
    groundtruth_map = decode_segmap(torch.argmax(groundtruth, dim=0))
    
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
    plt.imshow(input_img)
    
    dice_score = dice_coef_multilabel(groundtruth,output)
    print(f'Dice score : {round(dice_score, 5)}')
    plt.tight_layout()
    plt.show(block=True)
    plt.savefig('/home/fiodice/project/plot_transform' + 'seg.png')



def dice_coef(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    smooth = 0.0001
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

# y_true, y_pred -> [N_CLASS, H, W]
def dice_coef_multilabel(y_true, y_pred, num_labels=5):
    #print(torch.min(y_true), torch.min(y_pred))
    #print(torch.max(y_true), torch.max(y_pred))
    if torch.is_tensor(y_true):
        y_true = y_true.detach().cpu().numpy()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.detach().cpu().numpy()

    dice_val = 0
    for i in range(num_labels):
        # dice score for single class
        dice_val += dice_coef(y_true[i,:,:,], y_pred[i,:,:,])
        #print(f'Dice Val {dice_val} for label {i}')
    # tot dice score is the mean of each dice score for each class
    return dice_val/num_labels


def eval_batch(data, output, labels, n_class=6):
    batch_size = len(data)
    dice_score, dice_batch = 0, 0
    for i in range(batch_size):
        dice_score = dice_coef_multilabel(labels[i], output[i], n_class)
        # view one sample for batch
        dice_batch += dice_score
    return dice_batch / batch_size


def train_val_dataset(dataset, val_split=0.2):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
    datasets = {}
    datasets['train'] = torch.utils.data.Subset(dataset, train_idx)
    datasets['val'] = torch.utils.data.Subset(dataset, val_idx)
    return datasets


def mean_std(dataset):
    loader = torch.utils.data.DataLoader(dataset,
                                            batch_size=1,
                                            shuffle=False,
                                            num_workers=0)
    nimages, mean, std  = 0, 0., 0.
    print('='*15, f'Calc mean and std','='*15)
    for batch, _ in tqdm(loader):
        # Rearrange batch to be the shape of [B, C, W * H]
        batch = batch.view(batch.size(0), batch.size(1), -1)
        # Update total number of images
        nimages += batch.size(0)
        # Compute mean and std here
        mean += batch.mean(2).sum(0) 
        std += batch.std(2).sum(0)
    mean /= nimages
    std /= nimages
    return mean, std


def normalize(dataset, mean, std):
    dataset_norm = []
    print('='*15, f'Normalizing dataset','='*15)
    for j in trange(len(dataset)):
        label = dataset[j][1]
        img = dataset[j][0]
        img_norm = (img - mean[0]) / (std[0])
        dataset_norm.append((img_norm,label))
    return dataset_norm


if __name__ == '__main__':
    path_data = '/home/fiodice/project/dataset_seg/data/'
    path_labels = '/home/fiodice/project/dataset_seg/mask/'

    transform = transforms.Compose([ transforms.Resize((512,512)),
                                     transforms.ToTensor()])

    whole_dataset = HeartSegmentation(path_data, path_labels,transform=transform, crossentropy_prepare = False)

    mean, std = mean_std(whole_dataset)
    #mean, std = [0.5884], [0.1927]
    print(mean, std)
    dataset = normalize(whole_dataset, mean, std)
    #mean, std = mean_std(dataset)
    #print(mean, std)

    datasets = train_val_dataset(dataset)

    train_loader = torch.utils.data.DataLoader(datasets['train'],
							batch_size = 2,
							shuffle = True,
							num_workers = 0)

    val_loader = torch.utils.data.DataLoader(datasets['val'],
							batch_size = 2,
							shuffle = False,
							num_workers = 0)

    model = UNet(in_channels=1, out_channels=5, init_features=32).to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, weight_decay = 0.0001, momentum = 0.8)
    criterion = torch.nn.BCEWithLogitsLoss().to(device)
    accumulation = 1

    for epoch in range(30):
        epoch_dice = 0.
        loss_cumul = AverageMeter('Loss', ':.4e')
        with tqdm(train_loader, unit="batch") as tepoch:
            counter = 0
            optimizer.zero_grad()
            for data, target in tepoch:
                data = data.to(device)
                target = target.to(device)
                out_mask = model(data)
				#loss = 0.5*criterion(out_mask,target)/(16.0)
                loss = criterion(out_mask, target)/(accumulation)
                loss.backward()
                counter += 1
                if counter % accumulation == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    counter = 0
                loss_cumul.update(loss.item(), data.size(0)*accumulation)
                tepoch.set_postfix(loss=loss_cumul.avg)
            epoch_dice += eval_batch(data, torch.nn.functional.softmax(out_mask, dim=1), target, n_class=5)

        loss_cumul = AverageMeter('Loss', ':.4e')
        print(f'Training dice score: {round(epoch_dice, 5)}')

        with tqdm(val_loader, unit="batch") as tepoch:
            with torch.no_grad():
                for data, target in tepoch:
                    data = data.to(device)
                    target = target.to(device)
                    out_mask = model(data)
                    loss = criterion(out_mask,target)/(accumulation)
                    counter += 1
                    if counter % accumulation == 0:
                        optimizer.zero_grad()
                        counter = 0
                    loss_cumul.update(loss.item(), data.size(0))

                    if epoch > 5 and epoch % 10 == 0:
                        random_sample = 0
                        show_samples(data[0].detach().cpu(), 
                                torch.nn.functional.softmax(out_mask, dim=1)[0].detach().cpu(), 
                                target[0].detach().cpu())

                    tepoch.set_postfix(loss=loss_cumul.avg)
    
    torch.save(model.state_dict(), "segmentation_model_fi.pt")
