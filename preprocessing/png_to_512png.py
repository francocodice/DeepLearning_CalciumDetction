from PIL import Image
from glob import glob

source_folder = '/home/fiodice/project/data_resize_2048/'
target_folder = '/home/fiodice/project/data_resize_512/'

for this_name in glob(source_folder+'*'):
    print("Converting ", this_name)
    im = Image.open(this_name)
    im.thumbnail((512, 512), Image.ANTIALIAS)
    im.save(target_folder+this_name.split('/')[-1], "PNG")