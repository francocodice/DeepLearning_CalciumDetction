import matplotlib.pyplot as plt
from pip import main
from pydicom import dcmread
from pydicom.data import get_testdata_file
import numpy as np
import os
from pydicom.pixel_data_handlers.util import apply_modality_lut


def checkdataset():
    path_dicom = "/home/fiodice/project/data_custom/"

    # Count file for folder data
    list_subfolders_with_paths = sorted([f.path for f in os.scandir(path_dicom) if f.is_dir()])
    for dir in list_subfolders_with_paths:
        if(len(os.listdir(dir + '/rx')) > 1):
            print(dir + ' : ',len(os.listdir(dir + '/rx')))


if __name__ == '__main__':
    Slope = ('0028','1053')     
    Intercept = ('0028','1052') 

    plot_dir = '/home/fiodice/project/plot_transform/sample'
    image_path = '/home/fiodice/project/data_custom/test/rx/IM-0434-0001.dcm'
    
    ds = dcmread(image_path)
    hu = apply_modality_lut(ds.pixel_array, ds)  
    

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
    plt.imshow(ds.pixel_array)

        
    ax1 = f.add_subplot(1, 2, 2)
    ax1.title.set_text('Apply LUT')
    ax1.grid(False)
    plt.imshow(hu)


    plt.savefig(plot_dir + '_s.png')

