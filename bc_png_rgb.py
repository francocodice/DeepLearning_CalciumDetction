import torch
import dataset
import copy
import dataset_png
from tqdm import tqdm
from utils import *
from model import *
from torch.optim.lr_scheduler import StepLR
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

SIZE_IMAGE = 1024
PATH_PLOT = '/home/fiodice/project/plot_training/'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

def get_transforms():
    train_transforms = torchvision.transforms.Compose([
        transforms.RandomRotation(degrees=10),
        transforms.RandomPerspective(distortion_scale=0.3, p=0.4),
        torchvision.transforms.Resize((246,246)),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((246,246)),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    return train_transforms, test_transform


def run(model, dataloader, criterion, optimizer, scheduler=None, phase='train'):
    epoch_loss, epoch_acc, samples_num = 0., 0., 0.
    true_labels, pred_labels, outputs_labels = [], [], []
    max_probs = None
    
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

    if phase == 'test':
        probs_outputs = F.softmax(torch.cat(outputs_labels), dim=1)
        max_probs , _ = torch.max(probs_outputs, 1)
    
    if max_probs is not None:
        return epoch_loss / len(dataloader), epoch_acc / samples_num, torch.cat(true_labels).numpy(), torch.cat(pred_labels).numpy(), max_probs.numpy()
    else:
        return epoch_loss / len(dataloader), epoch_acc / samples_num, torch.cat(true_labels).numpy(), torch.cat(pred_labels).numpy(), _


def imshow(id, img):
    f = plt.figure()
    plt.imshow((img.cpu().permute(1, 2, 0).numpy() + mean) * std,cmap=plt.cm.gray)        
    plt.savefig(PATH_PLOT + 'error_'  + str(id) + '.png')
    plt.close()


def show_wrong_classified(best_pred_labels, true_labels, test_set):
    res = best_pred_labels == true_labels
    for id, (data, _) in enumerate(test_set):
        if res[id] == False:
            imshow(id, data)


if __name__ == '__main__':
    path_train_data = '/home/fiodice/project/dataset_png/train/'
    path_test_data = '/home/fiodice/project/dataset_png/test/'
    path_labels = '/home/fiodice/project/dataset/site.db'

    batchsize = 8

    train_t, test_t = get_transforms()


    train_set = dataset_png.CalciumDetectionPNG(path_train_data, path_labels, transform=train_t)
    test_set = dataset_png.CalciumDetectionPNG(path_test_data, path_labels, transform=test_t)

    #model = torchvision.models.resnet18(pretrained=True)
    model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', pretrained=True)
    for param in model.parameters():
        param.requires_grad = False

    model.classifier = nn.Linear(1024, 2)
    model = model.to(device)


    train_loader = torch.utils.data.DataLoader(train_set,
                                            batch_size=batchsize,
                                            shuffle=False,
                                            num_workers=0)

    test_loader = torch.utils.data.DataLoader(test_set,
                                            batch_size=batchsize,
                                            shuffle=False,
                                            num_workers=0)

    #show_distribution(train_loader, 'train', PATH_PLOT)
    #show_distribution(test_loader, 'test')

    best_model = None
    best_test_loss = 0.
    best_test_acc = 0.
    best_pred_labels = []
    best_prob_labels = []
    true_labels = []
    pred_labels = []
    test_acc = 0.
    test_loss = 0.

    #criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([1, 1.2], device=device))
    #loss = F.binary_cross_entropy_with_logits(preds, targets, reduction='none')
    criterion = torch.nn.CrossEntropyLoss()

    lr = 0.001
    weight_decay = 0.0001
    momentum = 0.9
    epochs = 20
    optimizer = torch.optim.SGD(model.classifier.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
    #scheduler = StepLR(optimizer, step_size=15, gamma=0.95)
    scheduler = None

    print(f'Criterion {criterion}, lr {lr}, weight_decay {weight_decay}, momentum : {momentum}, batchsize : {batchsize}')
    
    train_losses,test_losses  = [], []
    for epoch in range(1, epochs+1):
        print('='*15, f'Epoch: {epoch}','='*15)

        train_loss, train_acc, _, _, _ = run(model, train_loader, criterion, optimizer, scheduler=scheduler)
        test_loss, test_acc, true_labels, pred_labels, max_probs = run(model, test_loader, criterion, optimizer,scheduler=scheduler, phase='test')

        print(f'\nTrain loss: {train_loss:.4f}, Train accuracy: {train_acc:.4f}')
        print(f'Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.4f}\n')

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        if best_model is None or (test_loss < best_test_loss):
            print('update model')
            best_model = copy.deepcopy(model)
            best_test_loss = test_loss
            best_test_acc = test_acc 
            best_pred_labels = pred_labels
            best_prob_labels = max_probs

    torch.save({'model': best_model.state_dict()}, f'calcium-detection-x-ray.pt')

    #save_cm(true_labels, best_pred_labels, path_plot)
    print(f'Best model test accuracy: {best_test_acc:.4f}')
    print(f'Best model test loss: {best_test_loss:.4f}')

    save_losses(train_losses, test_losses, PATH_PLOT)
    save_cm(true_labels, best_pred_labels, PATH_PLOT)
    save_roc_curve(true_labels, best_prob_labels, PATH_PLOT)

    show_wrong_classified(best_pred_labels, true_labels, test_set)