
from PIL import Image
import os
import pydicom
import numpy as np
import sqlite3
from pydicom.pixel_data_handlers.util import apply_windowing
from utils import *
from PIL import Image
import skimage

def plot_preprocessing(path, img, img_proc, name):
    f = plt.figure()
                        
    ax1 = f.add_subplot(1, 2, 1)
    ax1.title.set_text('IMG')
    ax1.grid(False)
    plt.imshow(img, cmap='gray')
                        
    ax2 = f.add_subplot(1, 2, 2)
    ax2.title.set_text('Processing')
    ax2.grid(False)
    plt.imshow(img_proc, cmap='gray')

    plt.tight_layout()
    plt.savefig(path + name)


if __name__ == '__main__':
    path_train_data = '/home/fiodice/project/dataset/'
    path_labels = '/home/fiodice/project/dataset/site.db'
    path_img_out = '/home/fiodice/project/data_resize/'

    # File of dicom image
    DCM_files = []
    for dir_name, sub_dir_list, file_list in os.walk(path_train_data):
        for filename in file_list:
            if ".dcm" in filename.lower():
                DCM_files.append(os.path.join(dir_name, filename))

    #print("Number of (.dcm) files =", len(DCM_files))


    for index,this_name in enumerate(DCM_files):
        print("Converting ", this_name)
        dimg = pydicom.dcmread(this_name, force=True)
        img16 = apply_windowing(dimg.pixel_array, dimg)   

        img_2048 = skimage.transform.resize(img16, (2048, 2048), preserve_range=True)
        data_reshaped = img_2048 - img_2048.min()

        scale_factor = 4095.0 / data_reshaped.max()
        data_clip = np.clip(4095.0 - data_reshaped * scale_factor, 0, 4095)
        data_to8 = (data_clip / 16.0).astype('uint8')

        img = ~data_to8 if dimg.PhotometricInterpretation == 'MONOCHROME2' else data_to8

        imgOut = Image.fromarray(img)
        out_name = dimg.PatientID + '.png'
        #plot_preprocessing(target_folder, img16, imgOut, out_name)

        imgOut.save(path_img_out+out_name)
