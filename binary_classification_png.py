import torch
import project.src.dataset_png as dataset_png
import seaborn as sns
from utils import *
import model
import copy
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from model import *
from torch.optim.lr_scheduler import StepLR
from transform import *

SIZE_IMAGE = 1024
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
path_plot = '/home/fiodice/project/plot_transform/aug'

def run(model, dataloader, criterion, optimizer, scheduler=None, phase='train'):
    epoch_loss, epoch_acc = 0., 0.
    samples_num  = 0.
    true_labels, pred_labels = [], []
    
    #index = 0
    for (data, labels) in tqdm(dataloader):
        data, labels = data.to(device), labels.to(device)
        
        #show_sample(path_plot,index, data[0].cpu())
        #index += 1

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
    path_train_data = '/home/fiodice/project/data_resize_2048/'
    path_labels = '/home/fiodice/project/dataset/site.db'
    #path_plot = '/home/fiodice/project/plot_transform/sample'
    path_model = '/home/fiodice/project/model/final.pt'

    transform = transforms.Compose([ transforms.Resize((1556,1556)),
                                     transforms.CenterCrop(SIZE_IMAGE),
                                     transforms.ToTensor()])

    # Normalization here decrese accuracy of the model and precision 
    # of mean, std overall the dataset.

    #cac_dataset = dataset.CalciumDetection(path_train_data, path_labels, transform)
    cac_dataset = dataset_png.CalciumDetectionSegmentationPNG(path_train_data, path_labels, transform)

    model = load_densenet(path_model)
    model.to(device)

    size_train = 0.80

    train_set, test_set = split_train_val(size_train, cac_dataset)

    mean, std = mean_std(train_set)
    print(mean, std)
    #mean, std = [0.592], [0.192]

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
    #lr = 1e-3
    lr = 0.001
    weight_decay = 0.0001
    #weight_decay = 1e-4

    momentum = 0.8
    epochs = 60
    optimizer = torch.optim.SGD(model.fc.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)

    #optimizer = torch.optim.Adam(model.fc.parameters(), lr=lr, betas=(0.8, 0.999), eps=1e-08, weight_decay=weight_decay)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=True, factor=0.05)
    scheduler = StepLR(optimizer, step_size=20, gamma=0.95)

    print(f'Criterion {criterion}, lr {lr}, weight_decay {weight_decay}, momentum : {momentum}, batchsize : {batchsize}')
    train_losses,test_losses  = [], []

    for epoch in range(1, epochs+1):
        print('='*15, f'Epoch: {epoch}','='*15)
        
        train_loss, train_acc, _, _ = run(model, train_loader, criterion, optimizer, scheduler=scheduler)
        test_loss, test_acc, true_labels, pred_labels = run(model, test_loader, criterion, optimizer,scheduler=scheduler, phase='test')
        
        print(f'Train loss: {train_loss}, Train accuracy: {train_acc}')
        print(f'Test loss: {test_loss}, Test accuracy: {test_acc}\n')
        
        train_losses.append(train_loss)
        test_losses.append(test_loss)
            
        if best_model is None or (test_loss < best_loss) or (test_acc > best_test_acc):
            best_model = copy.deepcopy(model)
            best_test_loss = test_loss
            best_test_acc = test_acc 
            best_pred_labels = pred_labels

    torch.save({'model': best_model.state_dict()}, f'calcium-detection-x-ray.pt')

    #evalute_model(train_losses, test_losses, path_plot)
    #save_cm(true_labels, best_pred_labels, path_plot)
    print(f'Best test loss: {best_test_loss}, Best test accuracy: {best_test_acc}')

    plt.figure(figsize=(16, 8))
    plt.plot(train_losses, label='Train loss')
    plt.plot(test_losses, label='Test loss')
    plt.legend()
    plt.savefig(path_plot  + 'res.png')
    plt.close()

    print(f'Best model test accuracy: {best_test_acc}')
    print(f'Best model test loss: {best_test_loss}')

    #print(f'True Labels {true_labels}')
    #print(f'Pred Labels {best_pred_labels}')

    #### Confusion Matrix ####
    cm = confusion_matrix(true_labels, best_pred_labels)
    print(cm)
    ax = sns.heatmap(cm, annot=True, fmt="d")
    hm = ax.get_figure()
    hm.savefig(path_plot)
    hm.clf()
    plt.close(hm)