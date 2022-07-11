
import sys
import numpy as np
import sqlite3
import torch
import matplotlib.pyplot as plt


PATH_PLOT = '/home/fiodice/project/plot_training/'

sys.path.insert(0, '/home/fiodice/project/src')


if __name__ == '__main__':
    #x=np.arange(0,10)
    y=np.array([218, 225, 233, 230, 228, 239])
    x=np.arange(start=0, stop=6)
    #diff = x -y
    up_error = [0,0,0,40,50,60]
    bottom_error = [90,90,20,0,0,0]
    # ridimensioniamo l'immagine
    plt.figure(figsize=(10,10))
    # assegniamo etichette agli assi
    plt.xlabel("Calcium score")
    plt.ylabel("Errore")
    # impostiamo il titolo del grafico
    plt.title("Error")
    # chiediamo di visualizzare la griglia
    plt.grid()
    # disegniamo due linee
    plt.errorbar(x,y,yerr=[bottom_error, up_error],fmt='v')
    plt.show()
    plt.savefig(PATH_PLOT  + 'error_cac2.png')
    plt.close()
