import matplotlib.pyplot as plt
from pip import main
from pydicom import dcmread
import pydicom
from pydicom.data import get_testdata_file
import numpy as np
import os
from pydicom.pixel_data_handlers.util import apply_modality_lut, apply_windowing



def windowing(img, window_center, window_width, intercept=0, slope=1):
    # To HU
    img = (img*slope + intercept)
    
    img_min = window_center - window_width//2 #minimum HU level
    img_max = window_center + window_width//2 #maximum HU level
    
    img[img < img_min] = img_min 
    img[img > img_max] = img_max 
    
    #print(f'Img_w MIN {img_min} and MAX {img_max}')

    return img

  
def windowing_param(data):
    w_param = [data[('0028','1050')].value, #window center
                data[('0028','1051')].value] #window width
                    
    return to_int(w_param[0]),to_int(w_param[1])

def to_int(x):
    if type(x) == pydicom.multival.MultiValue: return int(x[0])
    else: return int(x)


def folders_file():
    path_dicom = "/home/fiodice/project/dataset/"
    
    # Doppia radiografia frontale: 326 -> IM-0003-1003.dcm, IM-0002-1002.dcm 
    #                               439 -> IM-0104-1001.dcm, IM-0105-1002.dcm

    # File of dicom image
    DCM_files = []
    for dir_name, sub_dir_list, file_list in os.walk(path_dicom):
        for filename in file_list:
            if ".dcm" in filename.lower():
                DCM_files.append(os.path.join(dir_name, filename))

    print("Number of (.dcm) files =", len(DCM_files))
    
    return DCM_files


if __name__ == '__main__':

    plot_dir = '/home/fiodice/project/plot_data/'
    image_path = '/home/fiodice/project/dataset/CAC_495/rx/IM-0248-0004.dcm'

    list_img = folders_file()

    for id, image_path in enumerate(list_img):

        if id % 1 == 0:
    
            ds = dcmread(image_path)
            w = apply_windowing(ds.pixel_array, ds)  

            #hu = apply_modality_lut(ds.pixel_array, ds)  
            #w_center, w_width = windowing_param(ds) 
            #img16 = windowing(hu, w_center, w_width)
            

            # Normal mode:
            print(ds[('0018','5101')].value)

            print(f"SOP Class........: {ds.SOPClassUID} ({ds.SOPClassUID.name})")
            print()

            pat_name = ds.PatientName
            display_name = pat_name.family_name + ", " + pat_name.given_name
            print(f"Patient's Name....................: {display_name}")
            print(f"Patient ID........................: {ds.PatientID}")
            print(f"Modality..........................: {ds.Modality}")
            print(f"PhotometricInterpretation.........: {ds.PhotometricInterpretation}")
            print(f"Study Date........................: {ds.StudyDate}")
            print(f"Image size........................: {ds.Rows} x {ds.Columns}")
            #print(f"Pixel Spacing.....................: {ds.PixelSpacing}")

            # use .get() if not sure the item exists, and want a default value if missing
            #print(f"Slice location...: {ds.get('SliceLocation', '(missing)')}")

            #plt.imshow(~ds.pixel_array, cmap=plt.cm.gray)
            #plt.savefig(image_path + '_inv.png')
            
            f = plt.figure()
            ax1 = f.add_subplot(1, 2, 1)
            ax1.title.set_text('Normal')
            ax1.grid(False)
            plt.imshow(ds.pixel_array,cmap=plt.cm.gray)        
            ax1 = f.add_subplot(1, 2, 2)
            ax1.title.set_text('Windowing')
            ax1.grid(False)
            plt.imshow(w,cmap=plt.cm.gray)
            plt.tight_layout()
            plt.savefig(plot_dir + str(ds.PatientName) + '_plot.png')
            plt.close()

