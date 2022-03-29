import sqlite3
import collections
import seaborn as sns
import pandas as pd 
import matplotlib.pyplot as plt

plot_dir = '/home/fiodice/project/plot_transform/sample'

def class_score_binary(cac_score):
    if int(cac_score) in range(0, 100):
        return 0
    else:
        return 1

# @TODO to complete

if __name__ == '__main__':
    labels_path = '/home/fiodice/project/data/calcium/site.db'
    plot_dir = '/home/fiodice/project/plot_transform/sample'

    conn = sqlite3.connect(labels_path)
    conn.row_factory = sqlite3.Row  
    cursor = conn.cursor()

    labels = [dict(row) for row in cursor.execute('SELECT * FROM patient').fetchall()]
    
    #ages = [2022 - int(row['birth'].split('-')[0]) for row in labels]
    #cac_scores = [int(row['cac_score']) for row in labels]
    labels = [class_score_binary(int(row['cac_score'])) for row in labels]
    count_labels = collections.OrderedDict(sorted(collections.Counter(labels).items()))

    val_samplesize = pd.DataFrame.from_dict(
        {'[0:100]': [count_labels[0]], 
        '[>100]': count_labels[1],
        })

    sns.barplot(data=val_samplesize).set_title('Labels distribution', fontsize=15)
    plt.savefig(plot_dir + 'analisi.png')

    # sesso, 
    # rx_date - ct_date
    

def view_loader(dataloader, msg):
    batch_labels = [label.tolist() for _, label in dataloader]
    label_flat_list = [item for sublist in batch_labels for item in sublist]
    count_labels = collections.OrderedDict(sorted(collections.Counter(label_flat_list).items()))
    print(f'For {msg} Labels {count_labels}')

    # count_data = pd.DataFrame.from_dict(
    #     {'0': [count_labels[0]], 
    #     '[0:100]': count_labels[1],
    #     '[100:400]': count_labels[2],
    #     '> 400': count_labels[3],
    #     })

    # print(f'CountData {count_data}')
    # sns.barplot(data=count_data).set_title(msg, fontsize=20)
    # plt.savefig(plot_dir + msg + 'analisi.png')
