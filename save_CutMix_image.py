import random
import math
from torchvision import transforms
import torch
import cv2
import numpy as np
from PIL import Image
import os
import re

def improve_image(img):
    img2array_RGB =  cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)
    img2array =cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2GRAY)
    ret,thresh1 = cv2.threshold(img2array,125,225,cv2.THRESH_BINARY_INV)
    kernel = np.ones((3,3), np.uint8) 
    dilation_img = cv2.dilate(thresh1, kernel, iterations = 1)
    gray_lap = cv2.Laplacian(dilation_img, cv2.CV_16S, ksize=3)
    dst = cv2.convertScaleAbs(gray_lap) # 轉回uint8

    contours, hierarchy = cv2.findContours(dst, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    mx = (0,0,0,0)      # biggest bounding box so far
    mx_area = 0
    for cont in contours:
        x, y, w, h = cv2.boundingRect(cont)
        area = w * h
        if area > mx_area:
            mx = x, y, w, h
            mx_area = area
    x,y,w,h = mx
    cv2.rectangle(img2array_RGB, (x, y), (x+w, y+h), (0, 0, 225), 0)

    image_cropped = img2array_RGB[y+1:y+h,x+1:x+w]
    image_cropped = Image.fromarray(image_cropped)
    h, w = image_cropped.size

    if h > w :
        w = h
    else :
        h = w
    
    if h <300 and w <300  :
        h = w = 380

    return h, w


object_class = []
file_image_name = []

input_file_list = 'C:/Users/MVCLAB/Desktop/tools/candle'
object_class = input_file_list.split('/')
output_file_path = 'C:/Users/MVCLAB/Desktop/tools'+'/'+ object_class[5] + '_CutMix_Saveimage'
# output_file_path = 'C:/Users/MVCLAB/Desktop/tools'+'/'+ object_class[5] + '_CutMix_Saveimage_improve'


if  not os.path.exists(output_file_path):
     os.makedirs(output_file_path)

for file in os.listdir(input_file_list):
        file_image_name.append(file)

image_count = len(file_image_name)

for image_name in file_image_name:
    img = Image.open(input_file_list +'/'+ image_name)
    resize_image = transforms.Resize([448,448])
    img = resize_image(img)

    # h, w = improve_image(img)

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

    stringName = re.sub(".JPG","",image_name)
    image_np = np.array(augmented)
    ConvertToNP_image = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    cv2.imwrite(os.path.join(output_file_path, f'{stringName}__cutmix_{object_class[5]}.JPG'),ConvertToNP_image)



    