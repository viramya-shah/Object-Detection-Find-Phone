from keras.models import load_model
import cv2
import sys
import warnings
warnings.filterwarnings('ignore')

img_loc = sys.argv[1]

model = load_model('my_model_log_loss_colab.h5')
print ("Model Loaded")

img_0 = cv2.imread(img_loc)
bilateral = cv2.bilateralFilter(img_0, 10, 180, 180)
edges = cv2.Canny(bilateral, 100, 200)
p = model.predict(edges.reshape(1,326, 490, 1))
print (p[0][0], p[0][1])