import torch
import torch.nn as nn
import dataset
import seaborn as sns
from utils import *
import model
import copy
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from model import *
import collections
from torch.optim.lr_scheduler import StepLR
from transform import *


##### CLASS NO MORE USED ###

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_densenet(path_model):
    model = HierarchicalResidual(encoder='densenet121')
    dict_model = torch.load(path_model)["model"]
    model.load_state_dict(dict_model)

    del model.fc1
    del model.fc2

    for param in model.parameters():
        param.requires_grad = False

    model.fc =  nn.Sequential(
            nn.Linear(1024, 32),
            nn.ReLU(),
            nn.Linear(32, 2))

    for param in model.fc.parameters():
        param.requires_grad = True
    
    return model


def view_loader(dataloader, msg):
    batch_labels = [label.tolist() for _, label in dataloader]
    label_flat_list = [item for sublist in batch_labels for item in sublist]
    count_labels = collections.OrderedDict(sorted(collections.Counter(label_flat_list).items()))
    print(f'For {msg} Labels {count_labels}')


def run_epoch(model, dataloader, criterion, optimizer, scheduler=None, phase='train'):
    epoch_loss, epoch_acc = 0., 0.
    samples_num  = 0.
    true_labels, pred_labels = [], []

    if scheduler is not None and phase == 'train':
        scheduler.step()
        print('LR:', scheduler.get_last_lr())
    

    for (data, labels) in tqdm(dataloader):
        data, labels = data.to(device), labels.to(device)
        
        optimizer.zero_grad()
        with torch.set_grad_enabled(phase == 'train'):
            outputs = model(data)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels).to(device)


        true_labels.append(labels.detach().cpu())
        pred_labels.append(preds.detach().cpu())
        
        if phase == 'train':
            loss.backward()
            optimizer.step()
        
        epoch_loss += loss.detach().cpu().item()
        epoch_acc += torch.sum(preds == labels.data)
        samples_num += len(labels)
    
    print()
    return epoch_loss / len(dataloader), epoch_acc / samples_num, torch.cat(true_labels).numpy(), torch.cat(pred_labels).numpy()


if __name__ == '__main__':
    path_train_data = '/home/fiodice/project/dataset/'
    path_labels = '/home/fiodice/project/dataset/site.db'
    path_plot = '/home/fiodice/project/plot_transform/sample'
    path_model = '/home/fiodice/project/model/final.pt'
    
    cac_dataset = dataset.CalciumDetection(path_train_data, path_labels)

    #mean, std = [0.596], [0.191]

    model = load_densenet(path_model)

    model.to(device)

    #size_train = 0.80
    #train_set, test_set = split_train_test(size_train, cac_dataset)

    #train_size = int(0.80 * len(cac_dataset))
    #test_size = len(cac_dataset) - train_size
    #train_set, test_set = torch.utils.data.random_split(cac_dataset, [train_size,test_size])

    valid_size = 0.2
    num_train = len(cac_dataset)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(valid_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]

    # define samplers for obtaining training and validation batches
    tng_dataset = torch.utils.data.Subset(cac_dataset, train_idx)
    test_dataset = torch.utils.data.Subset(cac_dataset, valid_idx)

    mean, std = [0.596], [0.191]
    train_transform, test_transform = get_transforms(img_size=1048,mean = mean, std = std)

    train_data_tf = dataset.TrasformDataset(tng_dataset, train_transform)
    test_data_tf = dataset.TrasformDataset(test_dataset, test_transform)

    batchsize = 4

    train_loader = torch.utils.data.DataLoader(train_data_tf,
                                            batch_size=batchsize,
                                            shuffle=False,
                                            num_workers=0)

    test_loader = torch.utils.data.DataLoader(test_data_tf,
                                            batch_size=batchsize,
                                            shuffle=False,
                                            num_workers=0)


    #mean, std = mean_std(train_data_tf)
    #print(mean, std)
    #mean, std = [0.596], [0.191]
    #train_set = normalize(train_set, mean, std)
    #test_set = normalize(test_set, mean, std)


    #view_loader(train_loader, 'train')
    #view_loader(test_loader, 'test')


    best_model = None
    best_loss = 1.
    best_test_loss = 0.
    best_test_acc = 0.
    best_pred_labels = []
    true_labels = []

    pred_labels = []
    test_acc = 0.
    test_loss = 0.

    criterion = torch.nn.CrossEntropyLoss( )
    #criterion = torch.nn.BCEWithLogitsLoss()

    lr = 1e-3
    weight_decay = 1e-4
    momentum=0.8
    epochs = 60

    #optimizer = torch.optim.Adam(model.fc.parameters(), lr=lr, betas=(0.8, 0.999), eps=1e-08, weight_decay=weight_decay)
    optimizer = torch.optim.SGD(model.fc.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
    
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=True, factor=0.05)
    #scheduler = StepLR(optimizer, step_size=10, gamma=0.95)

    print(f'Criterion {criterion}, lr {lr}, weight_decay {weight_decay}, momentum : {momentum}, batchsize : {batchsize}')

    train_losses,test_losses  = [], []

    for epoch in range(1, epochs+1):
        print('='*15, f'Epoch: {epoch}','='*15)
        
        train_loss, train_acc, _, _ = run_epoch(model, train_loader, criterion, optimizer)
        test_loss, test_acc, true_labels, pred_labels = run_epoch(model, test_loader, criterion, optimizer, phase='test')
        
        print(f'Train loss: {train_loss}, Train accuracy: {train_acc}')
        print(f'Test loss: {test_loss}, Test accuracy: {test_acc}')
        print()
        
        train_losses.append(train_loss)
        test_losses.append(test_loss)
            
        if best_model is None or test_loss < best_loss:
            best_model = copy.deepcopy(model)
            best_test_loss = test_loss
            best_test_acc = test_acc 
            best_pred_labels = pred_labels


    #evalute_model(train_losses, test_losses, path_plot)
    #save_cm(true_labels, best_pred_labels, path_plot)


    plt.figure(figsize=(16, 8))
    plt.plot(train_losses, label='Train loss')
    plt.plot(test_losses, label='Test loss')
    plt.legend()
    plt.savefig(path_plot  + 'res.png')

    print(f'Best model test accuracy: {best_test_acc}')
    print(f'Best model test loss: {best_test_loss}')

    #print(f'True Labels {true_labels}')
    #print(f'Pred Labels {best_pred_labels}')

    #### Confusion Matrix ####
    cm = confusion_matrix(true_labels, best_pred_labels)
    ax = sns.heatmap(cm, annot=True, fmt="d")
    hm = ax.get_figure()
    hm.savefig(path_plot)
    hm.clf()
    plt.close(hm)
    print(cm)
