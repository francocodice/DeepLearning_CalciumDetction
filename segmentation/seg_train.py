
import torch
import numpy as np
from pydicom.pixel_data_handlers.util import apply_modality_lut, apply_windowing
from torchvision import transforms
from dataset_seg import HeartSegmentation
from sklearn.model_selection import train_test_split
from model import UNet
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
from metrics import *
from visualize import *
import copy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

path_plot = '/home/fiodice/project/plot_transform/'


def train_val_dataset(dataset, val_split=0.20):
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



def run_epoch(model, dataloader, criterion, optimizer, epoch, phase='train'):
    size = len(dataloader)
    epoch_loss = 0.
    epoch_dice = 0.
    batch_num = 0.

    for batch_idx, (data, labels) in enumerate(dataloader):
        data, labels = data.to(device), labels.to(device)

        optimizer.zero_grad()
        with torch.set_grad_enabled(phase == 'train'):
            outputs = model(data)
            # output in range (0,1) for evaluating
            pred = torch.nn.functional.softmax(outputs, dim=1)
            # Only for cross Entropy
            #y_true = (torch.argmax(labels, dim=1, keepdim=True).type(torch.long)).squeeze(1)
            loss = criterion(outputs, labels).to(device)

        if phase == 'train':
            loss.backward()
            optimizer.step()

        print(f'\r{phase} batch [{batch_idx+1}/{size}]', end='', flush=True)
        epoch_loss += loss.detach().cpu().item()
        epoch_dice += eval_batch(data, pred, labels, n_class=3)
        batch_num += 1
        
        if epoch % 5 == 0 and phase=='test':
            show_samples(data[0].detach().cpu(), 
                    pred[0].detach().cpu(), 
                    labels[0].detach().cpu(),
                    batch_idx)

    print()
    return epoch_loss / batch_num, epoch_dice / batch_num


if __name__ == '__main__':
    path_data = '/home/fiodice/project/dataset_seg/data/'
    path_labels = '/home/fiodice/project/dataset_seg/mask/'

    #transform = transforms.Compose([ transforms.Resize((512,512)),
    #                                 transforms.ToTensor()])

    t_transforms = transforms.Compose([
            #transforms.RandomAffine(0, translate=(0, 0.2), scale=(1, 1.20)),
            transforms.Resize((512,512)),
            transforms.ToTensor()
    ])

    whole_dataset = HeartSegmentation(path_data, path_labels,transform=t_transforms, crossentropy_prepare = False)

    mean, std = mean_std(whole_dataset)
    #mean, std = [0.5884], [0.1927]
    print(f'\n Mean {mean} std {std} before normalization')
    dataset = normalize(whole_dataset, mean, std)
    #mean, std = mean_std(dataset)
    #print(f'\n Mean {mean} std {std} after normalization')

    datasets = train_val_dataset(dataset)

    batch_size = 8

    train_loader = torch.utils.data.DataLoader(datasets['train'],
							batch_size = batch_size,
							shuffle = True,
							num_workers = 0)

    test_loader = torch.utils.data.DataLoader(datasets['val'],
							batch_size = batch_size,
							shuffle = False,
							num_workers = 0)

    model = UNet(in_channels=1, out_channels=4, init_features=32).to(device)
    
    lr=0.08
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08)
    #optimizer = torch.optim.SGD(model.parameters(), lr=0.1, weight_decay = 0.0001, momentum = 0.8)
    criterion = torch.nn.BCEWithLogitsLoss().to(device)

    epochs = 90

    best_model = None
    best_loss = 2.
    best_test_loss = 0.
    best_test_dice = 0.

    train_losses = []
    val_losses = []
    test_losses = []

    for epoch in range(epochs+1):
        print('='*15, f'Epoch: {epoch}','='*15)

        train_loss, train_dice = run_epoch(model, train_loader, criterion, optimizer, epoch)
        test_loss, test_dice = run_epoch(model, test_loader, criterion, optimizer, epoch, phase='test')
        
        if epoch % 2 == 0:
            print(f'Train loss: {round(train_loss, 5)}, Training dice score: {round(train_dice, 5)}')
            print(f'Test loss: {round(test_loss, 5)}, Test dice score: {round(test_dice, 5)}')
            
        print()
        train_losses.append(train_loss)
        test_losses.append(test_loss)

        if best_model is None or test_loss < best_loss:
            best_model = copy.deepcopy(model)
            best_loss = test_loss
            best_test_loss = test_loss
            best_test_dice = test_dice
    
    torch.save(best_model.state_dict(), "segmentation_model_fi.pt")

    plt.figure(figsize=(18, 6))
    plt.plot(train_losses, label='Train loss')
    plt.plot(val_losses, label='Val loss')
    plt.plot(test_losses, label='Test loss')
    plt.legend()
    plt.savefig(path_plot  + 'loss.png')
    plt.close()

    print(f'Best model test dice: {best_test_dice}')
    print(f'Best model test loss: {best_test_loss}')
