
from matplotlib.pyplot import axes, ylabel
import numpy as np
from utils import *
import torchvision.transforms as transforms
import dataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler

PATH_PLOT = '/home/fiodice/project/plot_training/'


if __name__ == '__main__':
    path_data = '/home/fiodice/project/dataset_split/train/'
    path_data = '/home/fiodice/project/dataset_split/test/'
    path_labels = '/home/fiodice/project/dataset/site.db'

    train = dataset.CalciumDetectionRegression(path_data, path_labels, transform=None)
    test = dataset.CalciumDetectionRegression(path_data, path_labels, transform=None)

    train_loader = torch.utils.data.DataLoader(train,
                            batch_size = 1,
                            shuffle = False,
                            num_workers = 0)

    test_loader = torch.utils.data.DataLoader(test,
                            batch_size = 1,
                            shuffle = False,
                            num_workers = 0)

    loaders = [train_loader, test_loader]
    scores = []

    for loader in loaders:
        for batch_idx, (data, labels) in enumerate(loader):
            scores.append(labels.numpy()[0])

    score = np.array(scores)
    print(f'Score : Min {score.min()} Max {score.max()} Mean {score.mean()} Std {score.std()}')

    # edit bins
    plt.figure()
    plt.rcParams.update({'figure.figsize':(7,5), 'figure.dpi':100})
    plt.hist(score, bins=[0, 20, 200, 600, 1000, 4000, 10000])
    plt.gca().set(title='Frequency Histogram', xlabel='Calcium score', ylabel='Count')
    plt.savefig(PATH_PLOT + 'cac_frequency.png')
    plt.close()

    scaler = StandardScaler() 
    scaler_min = MinMaxScaler() 
    data_scaled_min = scaler_min.fit_transform(score.reshape(-1, 1))
    print(f'Score standardize: Min {data_scaled_min.min()} Max {data_scaled_min.max()} Mean {data_scaled_min.mean()} Std {data_scaled_min.std()}')

    mean = 0.12707100591715975
    std = 0.2166300882928299
    data_norm = (data_scaled_min - mean)/std

    print(f'Min {data_norm.min()} Max {data_norm.max()} Mean {data_norm.mean()} Std {data_norm.std()}')

    plt.figure()
    plt.rcParams.update({'figure.figsize':(7,5), 'figure.dpi':100})
    plt.hist(data_norm, bins=[0, 0.2, 0.4, 0.6, 0.8, 1])
    plt.gca().set(title='Frequency Histogram', xlabel='Calcium score', ylabel='Count')
    plt.savefig(PATH_PLOT + 'hist_scaled.png')
    plt.close()