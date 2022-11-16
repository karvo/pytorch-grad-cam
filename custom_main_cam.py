from cam import main
import os, time, shutil
import torch
import shutil
import torch.nn as nn
from torchvision import models, transforms
from custom_functions import load_model



model, model_filename, dst_path = load_model()
model.eval()

input_folder = 'custom_datapoints'
input_path = './' + input_folder + '/'
dirs = os.listdir(input_path)
print(dirs)

for d in dirs:
    print("Current directory",os.getcwd())
    print("path: ", input_path+d)
    if not os.path.isdir(input_path + d):
        continue
    files = os.listdir(input_path + d)
    total_files = len(files)
    file_counter = 0
    output_path = './Results/' + d + '/'
    if not os.path.isdir(output_path):
        os.makedirs(output_path)
    pre_existing_result_dirs = os.listdir(output_path)

    
    
    for imgname in files:
        if imgname.endswith('JPEG') or imgname.endswith('jpg') or imgname.endswith('png'):

            if not os.path.exists(output_path +imgname):
                shutil.copy(input_path+d+'/'+imgname, output_path)

            input_img = input_path + d + '/' + imgname
            print('imgname:', imgname)
            imgprefix = imgname.split('.')[0]
            print(imgprefix)
            
            # check if this file is already processes (results exist)
            
            if imgname in pre_existing_result_dirs:
                print('skipping')
                continue

            main(model, input_img, output_path, imgname)


src_path = "./Results/"
unique_id = time.strftime("%Y%m%d_%H%M%S")
dst_path = dst_path + model_filename + "/Results/" + unique_id  + "grad_cam" + "/"
shutil.copytree(src_path, dst_path)