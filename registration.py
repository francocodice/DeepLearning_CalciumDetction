import pydicom
import dataset
import utils
import skimage
from dataset import *
from PIL import Image

import pyelastix

PATH_PLOT = '/home/fiodice/project/plot_training/'

pyelastix.EXES = "/home/fiodice/.local/bin/elastix" 


def registration(input, reference):
	params = pyelastix.get_default_params(type='RIGID')
	params.NumberOfResolutions = 8
	params.AutomaticTransformInitialization = True
	params.AutomaticScalesEstimation = False
	params.NumberOfHistogramBins = 64
	params.MaximumStepLength = 5.0
	params.MaximumNumberOfIterations = 2000

	registered, field = pyelastix.register(input, reference, params)
	return registered

def dicom_img_c(path):
    dimg = dcm.dcmread(path, force=True)
    img16 = apply_windowing(dimg.pixel_array, dimg)   
    img8 = convert(img16, 0, 255, np.uint8)
    return ~img8 if dimg.PhotometricInterpretation == 'MONOCHROME1' else img8

if __name__ == '__main__':
    path_ref = '/home/fiodice/project/dataset/CAC_087/rx/IM-0001-0001-0001.dcm'
    path_dataset = '/home/fiodice/project/dataset/'

    ref_image = dicom_img_c(path = path_ref)
    ref_img_2048 = skimage.transform.resize(ref_image, (2048, 2048), preserve_range=True)

    DCM_files = []
    for dir_name, sub_dir_list, file_list in os.walk(path_dataset):
        for filename in file_list:
            if ".dcm" in filename.lower():
                DCM_files.append(os.path.join(dir_name, filename))

    for index, path in enumerate(DCM_files[:5]):
        print("Registration ", path)
        img = dicom_img_c(path)
        img_2048 = skimage.transform.resize(img, (2048, 2048), preserve_range=True)
        img_r = registration(img_2048, ref_img_2048)
        show([img_2048, ref_img_2048, img_r], str(index) + '_registration', PATH_PLOT)

