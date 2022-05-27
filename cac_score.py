
import dataset
from matplotlib.pyplot import axes, ylabel
import numpy as np
from utils import *
import torchvision.transforms as transforms


PATH_PLOT = '/home/fiodice/project/plot_training/'



def class_score_binary(cac_score):
    if int(cac_score) in range(0, 11):
        return 0
    else:
        return 1


if __name__ == '__main__':
    #th_norm = (100 - 622.3462) / 820.8487
    #th = np.log(th_norm + 1)
    #print(f'TH norm log {th}')
    #th_log = np.log(100 + 1)
    #th = norm_log(th_log)
    #print(f'TH log norm {th}')

    path_data = '/home/fiodice/project/dataset/'
    path_labels = '/home/fiodice/project/dataset/labels_new.db'

    train = dataset.CalciumDetectionRegression(path_data, path_labels, transform=None)
    #test = dataset.CalciumDetectionRegression(path_data, path_labels, transform=None)

    train_loader = torch.utils.data.DataLoader(train,
                            batch_size = 1,
                            shuffle = False,
                            num_workers = 0)

    # test_loader = torch.utils.data.DataLoader(test,
    #                         batch_size = 1,
    #                         shuffle = False,
    #                         num_workers = 0)

    loaders = [train_loader]
    scores = []

    for loader in loaders:
        for batch_idx, (data, labels) in enumerate(loader):
            scores.append(labels.numpy()[0])

    # score = np.clip(np.array(scores), 0, 2000)
    #log_score = np.log(score + 1)
    #norm_score = (log_score - 3.8316) / 3.5604
    # norm_score = (score - score.mean())/score.std()
    # log_norm_score = np.log(norm_score + 1)

    # norm_log_score = np.log(np.clip(np.array(scores), 0, 2000) + 1)
    # norm_log_score = (norm_log_score - norm_log_score.mean()) / norm_log_score.std()

    # print(f'Score : Min {score.min():.4f} Max {score.max():.4f} Mean {score.mean():.4f} Std {score.std():.4f}')
    # print(f'Log Norm Score : Min {log_norm_score.min():.4f} Max {log_norm_score.max():.4f} Mean {log_norm_score.mean():.4f} Std {log_norm_score.std():.4f}')
    # print(f'Norm Log Score : Min {norm_log_score.min():.4f} Max {norm_log_score.max():.4f} Mean {norm_log_score.mean():.4f} Std {norm_log_score.std():.4f}')

    plt.figure()
    plt.rcParams.update({'figure.figsize':(7,5), 'figure.dpi':100})
    plt.hist(scores, bins=int(180/1))
    plt.gca().set(title='Frequency Histogram of STD CAC score', xlabel='Calcium score', ylabel='Count')
    plt.savefig(PATH_PLOT + 'cac_frequency.png')
    plt.close()

    labels = []
    for s in scores:
        class_ref = class_score_binary(int(s))
        labels.append(class_ref)

    #labels = [class_score_binary(int(row['cac_score'])) for row in scores]
    count_labels = collections.OrderedDict(sorted(collections.Counter(labels).items()))

    val_samplesize = pd.DataFrame.from_dict(
        {'[0:10]': [count_labels[0]], 
          '[>10]': count_labels[1],
        })

    sns.barplot(data=val_samplesize).set_title('Labels distribution', fontsize=15)
    plt.savefig(PATH_PLOT + 'analisi2.png')

    #plt.figure()
    #plt.rcParams.update({'figure.figsize':(7,5), 'figure.dpi':100})
    #plt.hist(norm_score, bins=int(180/1))
    #plt.gca().set(title='Frequency Histogram of Log Clip CAC score', xlabel='calcium score', ylabel='Count')
    #plt.savefig(PATH_PLOT + 'cac_frequency_norm.png')
    #plt.close()

# if __name__ == '__main__':
#     path_data = '/home/fiodice/project/dataset_split/train/'
#     path_data = '/home/fiodice/project/dataset_split/test/'
#     path_labels = '/home/fiodice/project/dataset/site.db'

#     train = dataset.CalciumDetectionRegression(path_data, path_labels, transform=None)
#     test = dataset.CalciumDetectionRegression(path_data, path_labels, transform=None)

#     train_loader = torch.utils.data.DataLoader(train,
#                             batch_size = 1,
#                             shuffle = False,
#                             num_workers = 0)

#     test_loader = torch.utils.data.DataLoader(test,
#                             batch_size = 1,
#                             shuffle = False,
#                             num_workers = 0)

#     loaders = [train_loader, test_loader]
#     scores = []

#     for loader in loaders:
#         for batch_idx, (data, labels) in enumerate(loader):
#             scores.append(labels.numpy()[0])

#     score = np.array(scores)
#     print(f'Score : Min {score.min():.4f} Max {score.max():.4f} Mean {score.mean():.4f} Std {score.std():.4f}')

#     # edit bins
#     plt.figure()
#     plt.rcParams.update({'figure.figsize':(7,5), 'figure.dpi':100})
#     plt.hist(score, bins=int(180/1))
#     plt.gca().set(title='Frequency Histogram of CAC score', xlabel='Calcium score', ylabel='Count')
#     plt.savefig(PATH_PLOT + 'cac_frequency.png')
#     plt.close()

#     scaler = StandardScaler() 
#     scaler_min_max = MinMaxScaler() 

#     data_scaled_min = scaler_min_max.fit_transform(score.reshape(-1, 1))
#     print(f'Score STD Min {data_scaled_min.min():.4f} Max {data_scaled_min.max():.4f} Mean {data_scaled_min.mean():.4f} Std {data_scaled_min.std():.4f}')

#     plt.figure()
#     plt.rcParams.update({'figure.figsize':(7,5), 'figure.dpi':100})
#     plt.hist(data_scaled_min, bins=int(180/1))
#     plt.gca().set(title='Frequency Histogram of STD CAC score', xlabel='Calcium score', ylabel='Count')
#     plt.savefig(PATH_PLOT + 'std_cac_frequency.png')
#     plt.close()


#     mean = 0.12707100591715975
#     std = 0.2166300882928299
#     data_norm = (data_scaled_min - mean)/std
#     data_norm2 = scaler.fit_transform(data_scaled_min)
#     print(f'Score NORM Min {data_norm.min():.4f} Max {data_norm.max():.4f} Mean {data_norm.mean():.4f} Std {data_norm.std():.4f}')
#     print(f'Score NORM V2 Min {data_norm.min():.4f} Max {data_norm.max():.4f} Mean {data_norm.mean():.4f} Std {data_norm.std():.4f}')

#     plt.figure()
#     plt.rcParams.update({'figure.figsize':(7,5), 'figure.dpi':100})
#     plt.hist(data_norm, bins=int(180/1))
#     plt.gca().set(title='Frequency Histogram of Norm CAC score', xlabel='Calcium score', ylabel='Count')
#     plt.savefig(PATH_PLOT + 'hist_scaled.png')
#     plt.close()