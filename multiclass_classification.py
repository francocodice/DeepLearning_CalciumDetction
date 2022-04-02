from pip import main
import torch
import torchvision
import dataset
import copy
import matplotlib.pyplot as plt
from utils import *
from project.src.transform import *
from tqdm import tqdm
from sklearn.metrics import confusion_matrix



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

path_plot = '/home/fiodice/project/plot_transform/sample'


def run_epoch(model, dataloader, criterion, optimizer, phase='train'):
    epoch_loss, epoch_acc = 0., 0.
    samples_num  = 0.
    true_labels, pred_labels = [], []
    
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


def load_model_pretrained(path_model):
    model = torchvision.models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(in_features=512, out_features=3)

    model.load_state_dict(torch.load(path_model)['model'])
    
    # Freeze weights of CNN
    for param in model.parameters():
        param.requires_grad = False
        
    # Add FNC layer 
    model.fc = torch.nn.Linear(in_features=512, out_features=4)
    for param in model.parameters():
        print(param.requires_grad)
        
    model.to(device)

    return model


def load_model_torch():
    model = torchvision.models.resnet18(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False

    model.fc = torch.nn.Linear(in_features=512, out_features=4)
    model.to(device)
    return model



if __name__ == '__main__':

    path_data = '/home/fiodice/project/data_custom/'
    path_test_data = '/home/fiodice/project/data_test/'
    path_train_data = '/home/fiodice/project/data_train/'
    path_labels = '/home/fiodice/project/data_train/site.db'
    path_model = '/home/fiodice/project/model/resnet18-mooney-chest-x-ray-best.pt'

    print(device.type)
    
    ####### Load Data ############
    t_dataset = dataset.Calcium_detection(path_train_data, path_labels)
    test_set = dataset.Calcium_detection(path_test_data, path_labels, tensor_transform())

    ####### Load Model ############
    #model = load_model_torch()
    model =load_model_pretrained(path_model)
    print(model)
    ####### Augmentation ############
    #train_dataset = augmentation.increase_dataset(t_dataset)

    ####### Normalization ############
    mean, std = mean_std(train_dataset)
    dataset_norm = normalize(train_dataset, mean, std)

    # Check norm
    print(f'Mean {mean} \nStd {std} before Normalization\n')
    mean, std = mean_std(dataset_norm)
    print(f'Mean {mean} \nStd {std} after Normalization\n')

    ######## Split train and valition #######
    train_size = int(0.85 * len(dataset_norm))
    validation_size = len(dataset_norm) - train_size
    train_dataset, validation_dataset = torch.utils.data.random_split(dataset_norm, [train_size,validation_size])

    ####### Loading data ##########
    batchsize = 4

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                            batch_size=batchsize,
                                            shuffle=False,
                                            num_workers=0)

    val_loader = torch.utils.data.DataLoader(validation_dataset,
                                            batch_size=batchsize,
                                            shuffle=False,
                                            num_workers=0)

    test_loader = torch.utils.data.DataLoader(test_set,
                                            batch_size=batchsize,
                                            shuffle=False,
                                            num_workers=0)

    #### View Class distribution ####

    show_distribution(train_loader, 'train')
    show_distribution(val_loader, 'val')
    show_distribution(test_loader, 'test')



    #### TRAINING ###    

    best_model = None
    best_loss = 1.5
    best_test_loss = 0.
    best_test_acc = 0.
    best_val_acc = 0.0
    best_pred_labels = []
    true_labels = []

    pred_labels = []
    test_acc = 0.
    test_loss = 0.

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.fc.parameters(), lr=1e-3, weight_decay=0.001, momentum=0.8)
    #lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=True, factor=0.05)


    train_losses = []
    val_losses = []
    test_losses = []

    for epoch in range(1, 40):
        print('='*15, f'Epoch: {epoch}','='*15)
        
        train_loss, train_acc, _, _ = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, _, _ = run_epoch(model, val_loader, criterion, optimizer, phase='val')
        #lr_scheduler.step(val_loss)
        test_loss, test_acc, true_labels, pred_labels = run_epoch(model, test_loader, criterion, optimizer, phase='test')
        
        print(f'Train loss: {train_loss}, Train accuracy: {train_acc}')
        print(f'Val loss: {val_loss}, Val accuracy: {val_acc}')
        print(f'Test loss: {test_loss}, Test accuracy: {test_acc}')
        print()
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        test_losses.append(test_loss)
            
        if best_model is None or (val_loss < best_loss):
            best_val_acc = val_acc
            best_model = copy.deepcopy(model)
            best_loss = val_loss
            best_test_loss = test_loss
            best_test_acc = test_acc 
            best_pred_labels = pred_labels


    plt.figure(figsize=(18, 8))
    plt.plot(train_losses, label='Train loss')
    plt.plot(val_losses, label='Val loss')
    plt.plot(test_losses, label='Test loss')
    plt.legend()
    plt.savefig(path_plot  + 'res.png')

    print(f'Best model test accuracy: {best_test_acc}')
    print(f'Best model test loss: {best_test_loss}')

    print(f'True Labels {true_labels}')
    print(f'Pred Labels {best_pred_labels}')

    #### Confusion Matrix ####
    cm = confusion_matrix(true_labels, best_pred_labels)
    print(cm)
