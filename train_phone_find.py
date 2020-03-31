''' 
    -> Importing necessary libraries
    -> ALL are downloadable via pip
'''
import numpy as np
import pandas as pd
import glob
import os
import math
import sys
import matplotlib.pyplot as plt
import cv2
import warnings

from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, LeakyReLU, ReLU
from keras.optimizers import RMSprop, Adam

from sklearn.model_selection import train_test_split

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
folder_loc = sys.argv[1] + '/'
label_text_loc = folder_loc + "labels.txt"
image_loc = folder_loc + "*.jpg"

''' Reading the labels.txt file and storing it in dataframe '''
labels = pd.read_csv(str(label_text_loc), sep = ' ', header = None)
labels.columns = ['file_name', 'x', 'y']
labels['location'] = str(folder_loc) + labels['file_name']

'''
    Here I am doing the following thing
        1. Reading the image
        2. Applying the bilateral filtering
        3. Apply canny edge detection
        4. Flipping the image in vertical as well horizantal directions
        5. Appeding the images to a list
'''
img_data, img_name = [], []
for file_name, file_loc in zip(labels['file_name'], labels['location']):

    img_0 = cv2.imread(file_loc)
    bilateral = cv2.bilateralFilter(img_0, 10, 180, 180)
    edges = cv2.Canny(bilateral, 100, 200)
    
    img_0 = edges
    img_flip_v = cv2.flip(img_0, 0)
    img_flip_h = cv2.flip(img_0, 1)
    img_flip_vh = cv2.flip(img_0, -1)

    img_data.append(img_0)
    img_data.append(img_flip_v)    
    img_data.append(img_flip_h)    
    img_data.append(img_flip_vh)
    
    img_name.append(file_name.split(".")[0] + "_1.jpg")
    img_name.append(file_name.split(".")[0] + "_2.jpg")    
    img_name.append(file_name.split(".")[0] + "_3.jpg")    
    img_name.append(file_name.split(".")[0] + "_4.jpg")    

''' Reshaping the input data '''
img_data = np.array(img_data)
img_data = np.reshape(img_data, (len(img_data), 326, 490, 1))

''' Adjusting the co-ordinates wrt to the flipped position'''
y = []
for f in img_name:
    lx = labels.loc[labels['file_name'] == (f.split("_")[0] + ".jpg"), 'x'].values
    ly = labels.loc[labels['file_name'] == (f.split("_")[0] + ".jpg"), 'y'].values

    if f.split(".")[0].endswith("2"):
        ly = 1 - ly
    
    if f.split(".")[0].endswith("3"):
        lx = 1 - lx
        
    if f.split(".")[0].endswith("4"):
        ly = 1 - ly
        lx = 1 - lx

    y.append([lx, ly])

y = np.array(y)
y = np.reshape(y, (len(y), 2))

alpha = 0.2
''' Custom model architecture '''
model_layers = [
    Conv2D(16, (3, 3), strides = 1, input_shape = (326, 490, 1), activation='relu'),
    Conv2D(16, kernel_size = (3, 3), strides = 1, activation='relu'),
    MaxPooling2D(pool_size = (2, 2)),
    
    Conv2D(32, kernel_size = (3, 3), strides = 1, activation='relu'),
    Conv2D(32, kernel_size = (3, 3), strides = 1, activation = 'relu'),
    MaxPooling2D(pool_size = (2 ,2)),
        
    Conv2D(64, kernel_size = (3, 3), strides = 1, activation='relu'),
    Conv2D(64, kernel_size = (3, 3), strides = 1, activation='relu'),
    MaxPooling2D(pool_size = (2, 2)),

    Conv2D(128, kernel_size = (3, 3), strides = 1, activation='relu'),
    Conv2D(128, kernel_size = (3, 3), strides = 1, activation='relu'),
    MaxPooling2D(pool_size =(2, 2)),

    Conv2D(256, kernel_size = (3, 3), strides = 1, activation='relu'),
    Conv2D(256, kernel_size =  (3, 3), strides = 1, activation='relu'),
    MaxPooling2D(pool_size = (2 , 2)),
    
    Conv2D(256, kernel_size = (3, 3), strides = 1, activation='relu'),
    Conv2D(256, kernel_size = (3, 3), strides = 1, activation='relu'),
    MaxPooling2D(pool_size = (2 , 2)),

    Flatten(), 

    Dense(1240, activation='relu'), 
    Dense(640, activation='relu'), 
    Dense(480, activation='relu'), 
    Dense(120, activation='relu'), 
    Dense(62, activation='relu'), 
    Dense(2, activation='sigmoid')
]

model = Sequential(model_layers)
# print (model.summary())

''' in case you want to directly load the model '''
# model = load_model('my_model_log_loss_colab.h5')

''' Model comping using Adam optimizer with squared log error '''
model.compile(
    loss = 'mean_squared_logarithmic_error',
    optimizer = Adam(),
    metrics = ['accuracy']
)

''' Splitting the data into test (to be used for validation purposes only) 
    and training datasets
'''
X_train, X_test, y_train, y_test = train_test_split(
    img_data, 
    y, 
    test_size=0.1, 
    random_state=42
)

X_train = np.reshape(X_train, (len(X_train), 326, 490, 1))
y_train = np.reshape(y_train, (len(y_train), 2))

''' Fitting the model '''
model.fit(
    X_train,
    y_train,
    epochs = 20,
    batch_size = 16,
    validation_data = (X_test, y_test)
)

''' Checking for x_error and y_error on the entire dataset '''
new_labels = pd.DataFrame(columns = ['file_name', 'x1', 'y1'])
for i_name, i_data, _ in zip(img_name, img_data, y):
    p = model.predict(i_data.reshape(1,326, 490, 1))
    new_labels = new_labels.append({'file_name': i_name, 'x1' : p[0][0], 'y1' : p[0][1]}, ignore_index = True)

''' 
    Making a dataframe with the actual and predicted values and error between them.
    Storing the dataframe as master_final_custom.xlsx
'''
new_labels['x'], new_labels['y'] = y[..., 0], y[..., 1]
new_labels['x_err'] = (new_labels['x1'] - new_labels['x']).apply(absolute_value)
new_labels['y_err'] = (new_labels['y1'] - new_labels['y']).apply(absolute_value)
new_labels['x_acc'] = new_labels['x_err'].apply(val)
new_labels['y_acc'] = new_labels['y_err'].apply(val)
new_labels.head()
new_labels.to_excel("./master_final_custom.xlsx", sheet_name = 'master')
model.save('./my_model_log_loss_colab.h5')