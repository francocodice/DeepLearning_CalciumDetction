import torch
import dataset
import model
import copy
from tqdm import tqdm
from model import *
from utils import *

SIZE_IMAGE = 1024
PATH_PLOT = '/home/fiodice/project/plot_training/'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def run(model, dataloader, criterion, optimizer, scheduler=None, phase='train'):
    epoch_loss, epoch_acc = 0., 0.
    samples_num  = 0.
    true_labels, pred_labels = [], []
    
    for (data, labels) in tqdm(dataloader):
        data, labels = data.to(device), labels.to(device)
        
        #show_sample(path_plot,index, data[0].cpu())

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
    
    if scheduler is not None and phase == 'train':
        scheduler.step()
        print('LR:', scheduler.get_last_lr())

    print()
    return epoch_loss / len(dataloader), epoch_acc / samples_num, torch.cat(true_labels).numpy(), torch.cat(pred_labels).numpy()


if __name__ == '__main__':
    path_train_data = '/home/fiodice/project/dataset/'
    path_labels = '/home/fiodice/project/dataset/labels_new.db'
    path_model = '/home/fiodice/project/model/final.pt'

    transform = torchvision.transforms.Compose([ torchvision.transforms.Resize((1248,1248)),
                                     torchvision.transforms.CenterCrop(SIZE_IMAGE),
                                     torchvision.transforms.ToTensor()])

    cac_dataset = dataset.CalciumDetection(path_train_data, path_labels, transform)

    # Best model on
    # Test set : 75%
    # Train set : 77%  

    model = load_densenet_mlp(path_model)
    model.to(device)

    size_train = 0.80

    train_set, test_set = split_train_val(size_train, cac_dataset)

    mean, std = mean_std(train_set)
    print(mean, std)
    #mean, std = [0.5458], [0.2584]

    train_set = normalize(train_set, mean, std)
    test_set = normalize(test_set, mean, std)

    batchsize = 4

    train_loader = torch.utils.data.DataLoader(train_set,
                                            batch_size=batchsize,
                                            shuffle=False,
                                            num_workers=0)

    test_loader = torch.utils.data.DataLoader(test_set,
                                            batch_size=batchsize,
                                            shuffle=False,
                                            num_workers=0)

    #show_distribution(train_loader, 'train')
    #show_distribution(test_loader, 'test')

    best_model = None
    best_loss = 1.
    best_test_loss = 0.
    best_test_acc = 0.
    best_pred_labels = []
    true_labels = []
    pred_labels = []
    test_acc = 0.
    test_loss = 0.

    #criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([1, 1.2], device=device))
    #criterion = torch.nn.BCEWithLogitsLoss()

    criterion = torch.nn.CrossEntropyLoss()
    lr = 0.001
    weight_decay = 0.0001
    momentum = 0.8
    epochs = 80
    optimizer = torch.optim.SGD(model.fc.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)

    #optimizer = torch.optim.Adam(model.fc.parameters(), lr=lr, betas=(0.8, 0.999), eps=1e-08, weight_decay=weight_decay)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=True, factor=0.05)

    print(f'Criterion {criterion}, lr {lr}, weight_decay {weight_decay}, momentum : {momentum}, batchsize : {batchsize}')
    train_losses,test_losses  = [], []

    for epoch in range(1, epochs+1):
        print('='*15, f'Epoch: {epoch}','='*15)
        
        train_loss, train_acc, _, _ = run(model, train_loader, criterion, optimizer)
        test_loss, test_acc, true_labels, pred_labels = run(model, test_loader, criterion, optimizer, phase='test')
        
        print(f'Train loss: {train_loss:.4f}, Train accuracy: {train_acc:.4f}')
        print(f'Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.4f}\n')
        
        train_losses.append(train_loss)
        test_losses.append(test_loss)
            
        if best_model is None or (test_loss < best_loss):
            best_model = copy.deepcopy(model)
            best_test_loss = test_loss
            best_test_acc = test_acc 
            best_pred_labels = pred_labels

    torch.save({'model': best_model.state_dict()}, f'calcium-detection-x-ray.pt')


    print(f'Best model test accuracy: {best_test_acc}')
    print(f'Best model test loss: {best_test_loss}')

    save_losses(train_losses, test_losses, best_test_acc, PATH_PLOT)
    save_cm(true_labels, best_pred_labels, PATH_PLOT)