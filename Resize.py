import random
import math
from torchvision import transforms
import torch
import cv2
import numpy as np
from PIL import Image
import os
import re

img_path = "C:/Users/MVCLAB/Desktop/DataSet/VisA_20220922/candle/test/good/0038.JPG"
img = Image.open(img_path)
resize_image = transforms.Resize([448,448])
img = resize_image(img)

image_np = np.array(img)
ConvertToNP_image = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
cv2.imwrite('output.JPG',ConvertToNP_image)
