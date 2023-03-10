import random
import math
from torchvision import transforms
import torch
import cv2
import numpy as np
from PIL import Image
import os
import re

file_image_name = []
input_file_list = 'C:/Users/MVCLAB/Desktop/tools/candle'
output_file_path = 'C:/Users/MVCLAB/Desktop/tools/CutMix_Saveimage'

for file in os.listdir(input_file_list):
        file_image_name.append(file)
        

image_count = len(file_image_name)



for image_name in file_image_name:
    img = Image.open(input_file_list +'/'+ image_name)

    h = img.size[0]
    w = img.size[1]

    ratio_area = random.uniform(0.02, 0.15) * w * h
    log_ratio = torch.log(torch.tensor((0.3, 1 / 0.3)))
    aspect = torch.exp(
        torch.empty(1).uniform_(log_ratio[0], log_ratio[1])
    ).item()

    cut_w = int(round(math.sqrt(ratio_area * aspect)))
    cut_h = int(round(math.sqrt(ratio_area / aspect)))

    from_location_h = int(random.uniform(0, h - cut_h))
    from_location_w = int(random.uniform(0, w - cut_w))

    box = [from_location_w, from_location_h, from_location_w + cut_w, from_location_h + cut_h]
    patch = img.crop(box) # In this crop image


    to_location_h = int(random.uniform(0, h - cut_h))
    to_location_w = int(random.uniform(0, w - cut_w))

    insert_box = [to_location_w, to_location_h, to_location_w + cut_w, to_location_h + cut_h]
    augmented = img.copy()
    augmented.paste(patch, insert_box) # Let patch paste in augmented with location insert_box
    # augmented.show()
    # save_data = cv2.imread(augmented)

    stringName = re.sub(".JPG","",image_name)
    image_np = np.array(augmented)
    ConvertToNP_image = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    cv2.imwrite(os.path.join(output_file_path, f'{stringName}__cutmix.jpg'),ConvertToNP_image)



