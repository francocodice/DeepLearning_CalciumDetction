import os
from os import listdir
import sys
from pydicom import dcmread
import matplotlib.pyplot as plt
#from preprocessing import windowing, windowing_param
from pydicom.pixel_data_handlers.util import apply_modality_lut, apply_windowing, convert_color_space
from PIL.Image import fromarray

slope = ('0028','1053')     
intercept = ('0028','1052') 

if __name__ == '__main__':
    # EVN 
    # print(sys.prefix)
    path_dicom = "/home/fiodice/project/data_only_new/"
    # # # Doppia radiografia frontale: 326, 439

    # # # File of dicom image
    DCM_files = []
    for dir_name, sub_dir_list, file_list in os.walk(path_dicom):
         for filename in file_list:
             if ".dcm" in filename.lower():
                 DCM_files.append(os.path.join(dir_name, filename))
    print("Number of (.dcm) files =", len(DCM_files))

    plot_dir = '/home/fiodice/project/plot_training/'
    
    for index, image_path in enumerate(DCM_files):

        img = dcmread(image_path)
            #hu = apply_modality_lut(img.pixel_array, img)  
            #window = apply_windowing(hu, img)

        f = plt.figure()

        ax1 = f.add_subplot(1, 1, 1)
        ax1.title.set_text('Normal')
        ax1.grid(False)
        plt.imshow(img.pixel_array)
        plt.tight_layout()
        plt.savefig(plot_dir + str(index) + '.png')
