import numpy as np
import pandas as pd
import glob
import cv2
import sys
import warnings
import math
from gluoncv import model_zoo, data, utils
warnings.filterwarnings('ignore')

net = model_zoo.get_model('ssd_512_resnet50_v1_coco', pretrained=True)
print ("Model loaded")

file_path = str(sys.argv[1])

x, img = data.transforms.presets.yolo.load_test(file_path, short = 512)
class_IDs, scores, bounding_boxs = net(x)

''' a, b, c and d are the corners of the box '''
a, b, c, d = bounding_boxs[0][0]
a = a.asscalar()
b = b.asscalar()
c = c.asscalar()
d = d.asscalar()

''' The models scales up the image to 512*770. Thus, normalising the data to our own input'''
image_shape = img.shape

a = (a/image_shape[1])*490
c = (c/image_shape[1])*490
b = (b/image_shape[0])*326
d = (d/image_shape[0])*326

print (int((a+c)/2)/490, int((b+d)/2)/326)