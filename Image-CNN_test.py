import numpy as np
import cv2
from keras.preprocessing import image
from keras.models import load_model
from sklearn import metrics
import os
from sklearn.metrics import roc_curve, auc
from text_classifier import fusion_result_prob

model = load_model('my_model2.h5')

test_labels = np.zeros(200)
test_labels[100:200] = 1


img_path = 'D:\\test_images'
img_path_list = os.listdir(img_path)
results=[]
results1=[]
for img in img_path_list:
  img = os.path.join(img_path,img)
  test_image = image.load_img(img, target_size=(64, 64))
  test_image1 = image.img_to_array(test_image)
  test_image2 = np.expand_dims(test_image1, axis=0)
  results.append(model.predict(test_image2))
  results1.append(model.predict(test_image2))
  if results[-1] >= 0.5:
    results[-1] = 1.0
  else:
    results[-1] = 0.0
for i in range(100):
  if results[i]>0:
    print(i+1)
print(results)


ac_score = metrics.accuracy_score(test_labels, results)
recall = metrics.recall_score(test_labels, results)
precision = metrics.precision_score(test_labels, results)
F1 = metrics.f1_score(test_labels, results)
print(ac_score)
print(recall)
print(precision)
print(F1)

# Tanh function based decision fusion
Tanh_fusion_results=[]
for i,j in results1,fusion_result_prob:
  Tanh_fusion_results.append(2.0*(1.0/(1.0+exp(-2*i))+1.0/(1.0+exp(-2*j))-1.0))
  if Tanh_fusion_results[-1] < 1:
    Tanh_fusion_results[-1] = Tanh_fusion_results[-1];
  else:
    Tanh_fusion_results[-1] = 1.0

