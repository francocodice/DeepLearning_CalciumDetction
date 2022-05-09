import torch
import dataset
import copy

from sklearn.model_selection import KFold
from torch.optim.lr_scheduler import StepLR

from utils import *
from model import *

PATH_PLOT = '/home/fiodice/project/plot_training/'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)


def get_transforms(img_size, crop, mean, std):
    train_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        #transforms.RandomRotation(degrees=15),
        #transforms.RandomPerspective(distortion_scale=0.3, p=0.3),
        transforms.CenterCrop(crop),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.CenterCrop(crop),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    
    return train_transforms, test_transform



def save_losses_fold(train_losses, test_losses, best_test_acc, fold, path_plot):
    plt.figure(figsize=(16, 8))
    plt.title(f'Best accuracy : {best_test_acc:.4f}')
    plt.plot(train_losses, label='Train loss')
    plt.plot(test_losses, label='Test loss')
    plt.legend()
    plt.savefig(path_plot  + 'losses_fold' + str(fold) + '.png')
    plt.close()


def save_cm_fold(true_labels, best_pred_labels, fold, path_plot):
    cm = confusion_matrix(true_labels, best_pred_labels)
    ax = sns.heatmap(cm, annot=True, fmt="d")
    hm = ax.get_figure()
    hm.savefig(path_plot + 'cm_fold' + str(fold) + '.png')
    hm.clf()
    plt.close(hm)


def local_copy(dataset):
    data = []
    print('='*15, f'Copying dataset','='*15)
    for j in trange(len(dataset)):
        label = dataset[j][1]
        img = dataset[j][0]
        data.append((img,label))
    return data


def reset_weights(m):
  '''
    Try resetting model weights to avoid
    weight leakage.
  '''
  for layer in m.children():
   if hasattr(layer, 'reset_parameters'):
    print(f'Reset trainable parameters of layer = {layer}')
    layer.reset_parameters()


def run(model, dataloader, criterion, optimizer, scheduler=None, phase='train'):
    epoch_loss, epoch_acc, samples_num = 0., 0., 0.
    true_labels, pred_labels, outputs_labels = [], [], []
    
    for (data, labels) in tqdm(dataloader):
        data, labels = data.to(device), labels.to(device)
        
        optimizer.zero_grad()
        with torch.set_grad_enabled(phase == 'train'):
            outputs = model(data)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels).to(device)

        true_labels.append(labels.detach().cpu())
        pred_labels.append(preds.detach().cpu())
        outputs_labels.append(outputs.detach().cpu())
        
        if phase == 'train':
            loss.backward()
            optimizer.step()
        
        epoch_loss += loss.detach().cpu().item()
        epoch_acc += torch.sum(preds == labels.data)
        samples_num += len(labels)

    if scheduler is not None and phase == 'train':
        scheduler.step()

    
    return epoch_loss / len(dataloader), epoch_acc / samples_num, torch.cat(true_labels).numpy(), torch.cat(pred_labels).numpy()

  
if __name__ == '__main__':
    path_data = '/home/fiodice/project/dataset/'
    path_labels = '/home/fiodice/project/dataset/site.db'
    path_model = '/home/fiodice/project/model/final.pt'

    seed = 42
    set_seed(seed)

    k_folds = 5
    epochs = 40
    batchsize = 4
    results = {}
    mean, std = [0.5024], [0.2898]

    transform, _ = get_transforms(img_size=1248, crop=1024, mean = mean, std = std)

    whole_dataset = dataset.CalciumDetection(path_data, path_labels, transform)
    whole_dataset = local_copy(whole_dataset)

    kfold = KFold(n_splits=k_folds, shuffle=True)

    print('--------------------------------')
    criterion = torch.nn.CrossEntropyLoss()

    for fold, (train_ids, test_ids) in enumerate(kfold.split(whole_dataset)):
        print(f'FOLD {fold}')
        print('--------------------------------')
    
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
        
        train_loader = torch.utils.data.DataLoader(
                        whole_dataset, 
                        batch_size=batchsize, sampler=train_subsampler)
        test_loader = torch.utils.data.DataLoader(
                        whole_dataset,
                        batch_size=batchsize, sampler=test_subsampler)
        
        model = load_densenet_mlp(path_model)
        model.to(device)
        #model.apply(reset_weights)

        best_model = None
        best_test_loss = 1.
        best_test_acc = 0.
        best_pred_labels = []
        best_prob_labels = []
        true_labels = []
        pred_labels = []
        test_acc = 0.
        test_loss = 0.
        
        lr = 0.001
        weight_decay = 0.0001
        momentum = 0.8
        optimizer = torch.optim.SGD(model.fc.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
        scheduler = StepLR(optimizer, step_size=15, gamma=0.1)
        
        train_losses,test_losses  = [], []

        for epoch in range(1, epochs+1):
            print('\n','='*20, f'Epoch: {epoch}','='*20,'\n')

            train_loss, train_acc, _, _ = run(model, train_loader, criterion, optimizer, scheduler=scheduler)
            test_loss, test_acc, true_labels, pred_labels = run(model, test_loader, criterion, optimizer,scheduler=scheduler, phase='test')

            print(f'\nTrain loss: {train_loss:.4f}, Train accuracy: {train_acc:.4f}')
            print(f'Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.4f}\n')

            train_losses.append(train_loss)
            test_losses.append(test_loss)

            if best_model is None or (test_acc > best_test_acc):
                best_model = copy.deepcopy(model)
                best_test_loss = test_loss
                best_test_acc = test_acc 
                best_pred_labels = pred_labels

        #torch.save({'model': best_model.state_dict()}, f'calcium-detection-x-ray-seed-{seed}-fold-{fold}.pt')

        print('Accuracy for fold %d: %d %%' % (fold, 100.0 * best_test_acc))
        print('--------------------------------')
        results[fold] = 100.0 * best_test_acc
        save_losses_fold(train_losses, test_losses, best_test_acc, fold, PATH_PLOT)
        save_cm_fold(true_labels, best_pred_labels, fold, PATH_PLOT)
        
    # Print fold results
    print(f'K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS')
    print('--------------------------------')
    sum = 0.0

    for key, value in results.items():
        print(f'Fold {key}: {value} %')
        sum += value

    print(f'Average: {sum/len(results.items())} %')