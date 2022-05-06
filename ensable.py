import os
import random
import dataset
import numpy as np

from utils import *
from model import *

PATH_MODELS = '/home/fiodice/project/cac_models/'
PATH_PLOT = '/home/fiodice/project/plot_training/'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def ensable_choise(outputs):
    res = np.empty([len(outputs[0])])
    for i in range(len(outputs[0])):
        output_i = [out[i].item() for out in outputs]
        res[i] = np.bincount(output_i).argmax()
    return res

if __name__ == '__main__':
    path_test_data = '/home/fiodice/project/dataset_split/test/'
    path_labels = '/home/fiodice/project/dataset/site.db'
    models = []

    for model_state_dict in os.listdir(PATH_MODELS):
        model = load_densenet_class(PATH_MODELS + model_state_dict)
        model.to(device)
        models.append(model)
    
    mean, std = [0.5024], [0.2898]
    batchsize = 4

    train_t, test_t = get_transforms(img_size=1248, crop=1024, mean = mean, std = std)
    test_set = dataset.CalciumDetection(path_test_data, path_labels, transform=test_t)
    test_loader = torch.utils.data.DataLoader(test_set,
                                            batch_size=batchsize,
                                            shuffle=False,
                                            num_workers=0)

    true_labels, pred_labels = [], []
    ensamble_preds = []
    acc, samples_num = 0., 0

    for (data, labels) in tqdm(test_loader):
        data, labels = data.to(device), labels.to(device)
        with torch.no_grad():
            outputs = []
            for i, model in enumerate(models):
                output = model(data)
                _, preds = torch.max(output, 1)
                #print(f'Output {preds} for model {i}')
                outputs.append(preds.cpu())
            #print(f'Current ensable_labels {ensable_labels}')
            true_labels.append(labels.detach().cpu())
            pred_labels.append(torch.tensor(ensable_choise(outputs)))

            acc += torch.sum(preds == labels.data)
            samples_num += len(labels)

    print(f'Model test accuracy: {acc/samples_num:.4f}')
    save_cm(torch.cat(true_labels).numpy(), torch.cat(pred_labels).numpy(), PATH_PLOT)