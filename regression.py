import torch
import dataset
import copy

from tqdm import tqdm
from utils import *
from model import *
from torch.optim.lr_scheduler import StepLR

SIZE_IMAGE = 1024
# [0;1]
#THRESHOLD_CAC_SCORE = 0.0102
# Norm
#THRESHOLD_CAC_SCORE = -0.5398
# Log 
#THRESHOLD_CAC_SCORE = 0.2200653063816593

# TH for Clip -> Log  -> Norm
#THRESHOLD_CAC_SCORE = 0.2895

# TH for Clip -> Log  -> Norm -> TanH
THRESHOLD_CAC_SCORE = 0.2729

# TH for Clip -> Norm -> Log
#THRESHOLD_CAC_SCORE =  -1.0115604970081193

PATH_PLOT = '/home/fiodice/project/plot_training/'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def mean_absolute_error(y_true, y_pred):
    return np.sum(np.abs(np.array(y_pred) - np.array(y_true)))


def to_class(continuos_values, labels):
    classes_labels = [0 if labels[i] <= THRESHOLD_CAC_SCORE else 1 for i in range(labels.size(dim=0))]
    output_labels = [0 if continuos_values[i] <= THRESHOLD_CAC_SCORE else 1 for i in range(continuos_values.size(dim=0))]
    return torch.tensor(output_labels), torch.tensor(classes_labels)


def show_wrong_classified(best_pred_labels, true_labels, test_set):
    res = best_pred_labels == true_labels
    for id, (data, label) in enumerate(test_set):
        if res[id] == False:
            plt.figure()
            plt.title(f'Wrong classification, correct is {label}')
            plt.imshow((data.cpu().permute(1, 2, 0).numpy() + mean) * std,cmap=plt.cm.gray)        
            plt.savefig(PATH_PLOT + 'error_'  + str(id) + '.png')
            plt.close()


def run(model, dataloader, criterion, optimizer, scheduler=None, phase='train'):
    epoch_loss, epoch_acc, samples_num = 0., 0., 0.
    true_labels, pred_labels = [], []
    run_abs = 0.,
    
    for (data, labels) in tqdm(dataloader):
        data, labels = data.to(device), labels.to(device)
        
        optimizer.zero_grad()
        with torch.set_grad_enabled(phase == 'train'):
            outputs = model(data)
            output_classes, labels_classes = to_class(outputs.detach().cpu(), labels.detach().cpu())
            loss = criterion(outputs.float(), labels.unsqueeze(dim=1).float()).to(device)

        true_labels.append(labels_classes)
        pred_labels.append(output_classes)
        
        if phase == 'train':
            loss.backward()
            optimizer.step()
        
        epoch_loss += loss.detach().cpu().item()
        epoch_acc += torch.sum(labels_classes == output_classes)
        samples_num += len(labels)
        run_abs += mean_absolute_error(labels.detach().cpu(), outputs.detach().cpu())

    if scheduler is not None and phase == 'train':
        scheduler.step()

    return epoch_loss / len(dataloader), epoch_acc / samples_num, torch.cat(true_labels).numpy(), torch.cat(pred_labels).numpy(), run_abs / samples_num
    
if __name__ == '__main__':
    path_train_data = '/home/fiodice/project/dataset_split/train/'
    path_test_data = '/home/fiodice/project/dataset_split/test/'
    path_labels = '/home/fiodice/project/dataset/site.db'
    path_model = '/home/fiodice/project/model/final.pt'

    set_seed(42)

    # Mean and Std of ChestXpert dataset
    mean, std = [0.5024], [0.2898]
    batchsize = 4

    train_t, test_t = get_transforms(img_size=1248, crop=1024, mean = mean, std = std)

    train_set = dataset.CalciumDetectionRegression(path_train_data, path_labels, transform=train_t)
    test_set = dataset.CalciumDetectionRegression(path_test_data, path_labels, transform=test_t)

    model = load_densenet_mse(path_model)
    model.to(device)


    train_loader = torch.utils.data.DataLoader(train_set,
                                            batch_size=batchsize,
                                            shuffle=False,
                                            num_workers=0)

    test_loader = torch.utils.data.DataLoader(test_set,
                                            batch_size=batchsize,
                                            shuffle=False,
                                            num_workers=0)

    #show_distribution(train_loader, 'train', PATH_PLOT)
    #show_distribution(test_loader, 'test', PATH_PLOT)

    best_model = None
    best_test_acc = 0.
    best_pred_labels = []
    best_prob_labels = []
    true_labels = []
    pred_labels = []
    test_acc = 0.
    test_loss = 0.

    #criterion = torch.nn.MSELoss()

    criterion = torch.nn.L1Loss()

    lr = 0.001
    weight_decay = 0.0001
    momentum = 0.8
    epochs = 35
    optimizer = torch.optim.SGD(model.fc.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
    
    scheduler = StepLR(optimizer, step_size=15, gamma=0.1)
    #scheduler = None

    #print(f'Criterion {criterion}, lr {lr}, weight_decay {weight_decay}, momentum : {momentum}, batchsize : {batchsize}')
    
    train_losses,test_losses  = [], []
    train_accs, test_accs = [],[]
    trains_abs, tests_abs = [], []

    for epoch in range(1, epochs+1):
        print('='*15, f'Epoch: {epoch}','='*15)

        train_loss, train_acc, _, _ ,train_abs = run(model, train_loader, criterion, optimizer, scheduler=scheduler)
        test_loss, test_acc, true_labels, pred_labels, test_abs = run(model, test_loader, criterion, optimizer,scheduler=scheduler, phase='test')

        print(f'\nTrain loss: {train_loss:.4f}, Train accuracy: {train_acc:.4f}')
        print(f'Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.4f}\n')

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        trains_abs.append(train_abs)
        tests_abs.append(test_abs)

        if best_model is None or (test_acc > best_test_acc):
            best_model = copy.deepcopy(model)
            best_test_acc = test_acc 
            best_pred_labels = pred_labels

    #torch.save({'model': best_model.state_dict()}, f'calcium-detection-x-ray.pt')
    print(f'Best model test accuracy: {best_test_acc:.4f}')
    #print(f'Best model test loss: {best_test_loss:.4f}')
    
    save_losses(train_losses, test_losses, best_test_acc, PATH_PLOT)
    save_metric(train_accs, test_accs, 'accs', PATH_PLOT)
    save_metric(trains_abs, tests_abs, 'abs', PATH_PLOT)
    save_cm(true_labels, best_pred_labels, PATH_PLOT)
    #show_wrong_classified(best_pred_labels, true_labels, test_set)