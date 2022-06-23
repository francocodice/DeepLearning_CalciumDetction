
import dataset
import numpy as np
from utils import *
from datetime import date

import sqlite3
import collections
import seaborn as sns
import pandas as pd 
import matplotlib.pyplot as plt

PATH_PLOT = '/home/fiodice/project/plot_analyses/'


def save_cac_distribution(loader):
    scores = []

    for (_, labels) in (loader):
        scores.append(labels.numpy()[0])

    plt.figure()
    plt.rcParams.update({'figure.figsize':(7,5), 'figure.dpi':100})
    plt.hist(scores, bins=int(180/1))
    plt.gca().set(title='Frequency Histogram CAC score', xlabel='Calcium score', ylabel='Count')
    plt.savefig(PATH_PLOT + 'cac_frequency.png')
    plt.close()

    score_clip = np.clip(scores, a_min=0, a_max=2000)
    #print(train_score_clip.mean(), train_score_clip.std())
    log_cac_score = np.log(score_clip + 0.001)

    plt.figure()
    plt.rcParams.update({'figure.figsize':(7,5), 'figure.dpi':100})
    plt.hist(log_cac_score, bins=int(180/1))
    plt.gca().set(title='Frequency Histogram Log CAC score', xlabel='Calcium score', ylabel='Count')
    plt.savefig(PATH_PLOT + 'cac_log_frequency.png')
    plt.close()
    

if __name__ == '__main__':
    path_data = '/home/fiodice/project/dataset/'
    path_labels = '/home/fiodice/project/dataset/labels_new.db'

    whole_dataset = dataset.CalciumDetectionRegression(path_data, path_labels, transform=None)

    loader = torch.utils.data.DataLoader(whole_dataset,
                            batch_size = 1,
                            shuffle = False,
                            num_workers = 0)

    # Print CAC distribution and LogCac distribution
    #save_cac_distribution(loader)

    conn = sqlite3.connect(path_labels)
    conn.row_factory = sqlite3.Row  
    cursor = conn.cursor()

    labels = [dict(row) for row in cursor.execute('SELECT * FROM patient').fetchall()]

    patients = []
    ages, rx_date = [], []
    sex_f = 0
    sex_cac = {'f' : 0, 'm':0}

    cac_scores = [int(row['cac_score']) for row in labels]

    for label in labels:
        patients.append(label['id'])
        cac_scores.append(int(label['cac_score']))   
        ages.append(date.today().year - int(label['birth'].split('-')[0]))
        #error on label rx_date yaer = 21
        if(int(label['rx_date'].split('-')[0]) != 21):
            rx_date.append(int(label['rx_date'].split('-')[0]))

        sex_cac[label['sex']] += 1

        if label['sex'] == 'f':
            sex_f += 1

    # 2 error in labels
    ages = np.array(ages)
    rx_date = np.array(rx_date)

    print('Pazienti totali ' + '.'*20 + f' {len(set(patients)) - 2}')
    print('Pazienti sesso F ' + '.'*20 + f' {sex_f -1 } ')
    print('Pazienti sesso M ' + '.'*20 + f' {len(set(patients)) - sex_f - 1}')
    print('Pazienti età media ' + '.'*20 + f' {ages.mean()}')
    print('Pazienti età minima ' + '.'*20 + f' {ages.min()}')
    print('Pazienti età massima ' + '.'*20 + f' {ages.max()}')

    print('Radiografia più recente ' + '.'*20 + f' {rx_date.min()}')
    print('Radiografia meno recente ' + '.'*20 + f' {rx_date.max()}')


    


