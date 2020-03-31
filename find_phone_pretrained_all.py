''' 
    -> Importing necessary libraries
    -> ALL are downloadable via pip
'''
import numpy as np
import pandas as pd
import glob
import cv2
import sys
import warnings
import math
from gluoncv import model_zoo, data, utils

warnings.filterwarnings("ignore")

'''
    -> Custom function: 
        Return absolute value.
    Used to check the accuracy of the prediction i.e. (x_pred - x_true) and (y_pred - y_true)
'''
def absolute_value(x):
    return math.fabs(x)

'''
    -> Custom function:
        Returns how much is the prediction deviated from ground truth
            1 signifies that the prediction is within 5 percent error
            2 signifies that the prediction is within 15 percent error
            3 signifies that the prediction is more than 15 percent error
'''
def val(x):
    if x <= 0.05:
        return 1
    elif x <= 0.15:
        return 2
    else:
        return 3

''' Fetching arguments from CLI '''
folder_loc = sys.argv[1]
label_text_loc = folder_loc + "/labels.txt"
image_loc = folder_loc + "/*.jpg"

''' Reading the labels.txt file and storing it in dataframe '''
labels = pd.read_csv(label_text_loc, sep = ' ', header = None)
labels.columns = ['file_name', 'x', 'y']
labels['location'] = str(folder_loc) + "/" + labels['file_name']
print ("Labels read")

''' Fetching the pretrained model '''
net = model_zoo.get_model('ssd_512_resnet50_v1_coco', pretrained=True)
print ("Model loaded")

''' Calculating the coordinates of phone from image '''
bounding_box_ssd_512_vgg = []
new_labels = pd.DataFrame(columns = ['file_name', 'x_pred', 'y_pred'])
for file_path in labels['location']:
    x, img = data.transforms.presets.yolo.load_test(file_path, short = 512)
    class_IDs, scores, bounding_boxs = net(x)
    bounding_box_ssd_512_vgg.append(bounding_boxs[0][0])

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
    
    new_labels = new_labels.append({'file_name' : file_path, 'x_pred': int((a+c)/2)/490, 'y_pred': int((b+d)/2)/326}, ignore_index = True)
    ax = utils.viz.plot_bbox(img, bounding_boxs[0], scores[0], class_IDs[0], class_names = net.classes)

print ("Coordinates predicted")

''' 
    Making a dataframe with the actual and predicted values and error between them.
    Storing the dataframe as master_final_pretrained.xlsx
'''
new_labels = pd.merge(new_labels, labels, how = 'inner', left_on='file_name', right_on='location')
new_labels = new_labels.drop(['file_name_y', 'location'], axis = 1)
new_labels['x_err'] = (new_labels['x_pred'] - new_labels['x']).apply(absolute_value)
new_labels['y_err'] = (new_labels['y_pred'] - new_labels['y']).apply(absolute_value)
new_labels['x_acc'] = new_labels['x_err'].apply(val)
new_labels['y_acc'] = new_labels['y_err'].apply(val)
new_labels.to_excel('master_final_pretrained.xlsx', sheet_name = 'master')
print ("Excel file made")