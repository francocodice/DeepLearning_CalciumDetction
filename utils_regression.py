import torch

from utils import *
from model import *
from utils_model import *

PATH_PLOT = '/home/fiodice/project/plot_training/cross_35/'
MAX_CAC_VAL = 2000


def cac_prediction_error(labels, outputs, mean, std, best_pred_labels, true_labels):
    # true_labels -> binary label
    # labels -> continuos label
    labels = np.exp((labels * std) + mean - 0.001)
    outputs = np.exp((outputs * std) + mean - 0.001)
    total_error, error_on_wrong_sample = [], []
    for i in range(len(labels)): 
        total_error.append(np.abs(outputs[i] - labels[i]))
        if (best_pred_labels[i] != true_labels[i]):
            error_on_wrong_sample.append(np.abs(outputs[i] - labels[i]))
            print(f'For patient mis classfied {i} .... CAC label : {labels[i]} CAC predicted : {outputs[i]}')

    total_error = np.array(total_error)
    error_on_wrong_sample = np.array(error_on_wrong_sample)
    print(f'Total error Average: {total_error.mean()} STD : {total_error.std()}')
    print(f'Error Average on misclassified sample : {error_on_wrong_sample.mean()} STD : {error_on_wrong_sample.std()}')



def viz_distr_data(loader, fold, phase):
    scores = []
    for (_, labels) in loader:
        scores.append(labels.numpy()[0])
    plt.figure()
    plt.rcParams.update({'figure.figsize':(7,5), 'figure.dpi':100})
    plt.hist(scores, bins=int(180/1))
    plt.gca().set(title='Frequency Histogram CAC score fold ' + str(fold), xlabel='Calcium score', ylabel='Count')
    plt.savefig(PATH_PLOT + str(phase) + '_cac_frequency_fold_' + str(fold) + '.png')
    plt.close()


def pre_process_label(mean, std, labels):
    train_score_clip = np.clip(labels.detach().cpu(), a_min=0, a_max=MAX_CAC_VAL)
    train_log_score = np.log(train_score_clip + 0.001)
    return (train_log_score - mean) / std


def mean_std_cac_score_log(loader):
    train_score = torch.cat([labels for (_, labels) in loader]).numpy()
    #print(train_score.mean(), train_score.std())
    train_score_clip = np.clip(train_score, a_min=0, a_max=MAX_CAC_VAL)
    #print(train_score_clip.mean(), train_score_clip.std())
    train_log_score = np.log(train_score_clip + 0.001)
    #print(train_log_score.mean(), train_log_score.std())
    return train_log_score.mean(), train_log_score.std()


def norm_labels(mean, std, labels):
    cac_clip = np.clip([labels],a_min=0, a_max=MAX_CAC_VAL)
    log_cac_score = np.log(cac_clip + 0.001)
    return (log_cac_score - mean) / std


# Icrease speed of training 
def local_copy(dataset):
    return [(dataset[j][0],dataset[j][1]) for j in range(len(dataset))]


def local_copy_str_kfold(dataset):
    data = [dataset[j][0] for j in range(len(dataset))]
    label = [dataset[j][1] for j in range(len(dataset))]
    return data, label


def mean_absolute_error(y_true, y_pred):
    return np.sum(np.abs(np.array(y_true) - np.array(y_pred)))/len(y_true)


def to_class(continuos_values, labels, th):
    classes_labels = [0 if labels[i] <= th else 1 for i in range(labels.size(dim=0))]
    output_labels = [0 if continuos_values[i] <= th else 1 for i in range(continuos_values.size(dim=0))]
    return torch.tensor(output_labels), torch.tensor(classes_labels)