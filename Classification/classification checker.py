
''' Code to Check the model performance on unused retinal images'''

import numpy as np
import pandas as pd
from keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
import tensorflow.keras.utils as image
import glob
from sklearn import metrics

#%% Segmentation classifier model

segmented_model = load_model('keras_model')
#%% Segmented Classifier Prediction
y_pred = []
y_true = []

img_path = 'Unused for Training/*.jpg'
for file in glob.glob(img_path):
    print(file)
    
    img = image.load_img(file)

    # Resize the image
    img = img.resize((180, 180))

    # Convert the image to a NumPy array
    img_arr = np.array(img)

    # Preprocess the image
    img_arr = preprocess_input(img_arr)
    
    #Get Predicion probabilities
    preds_prob = segmented_model.predict(np.array([img_arr]))
    
    #Prediction
    pred = np.argmax(preds_prob, axis = 1)
    
    #Ground Truth
    check = glob.glob('Unused for Training/*0.jpg')
    if file in check:
        true = 1
    else:
        true = 0
        
    y_pred.append(pred[0])
    y_true.append(true)
    



#%%

segmented_cm = metrics.confusion_matrix(y_true, y_pred)
accuracy = metrics.accuracy_score(y_true, y_pred)
balanced_accuracy = metrics.balanced_accuracy_score(y_true, y_pred)
precision = metrics.precision_score(y_true, y_pred)
recall = metrics.recall_score(y_true, y_pred)
f1 = metrics.f1_score(y_true, y_pred)

dictionary = {'Accuracy': [accuracy], 'Balanced Accuracy': [balanced_accuracy],
              'Precision': [precision], 
              'Recall': [recall], 'f1 Score': [f1]}
segmented_metrics = pd.DataFrame(data = dictionary)

#%% Not Segmented Model
normal_model = load_model('comparison_model')

#%%
y_pred = []
y_true = []

img_path = 'Unused unmasked/*.jpg'
for file in glob.glob(img_path):
    print(file)
    
    img = image.load_img(file)

    # Resize the image
    img = img.resize((180, 180))

    # Convert the image to a NumPy array
    img_arr = np.array(img)

    # Preprocess the image
    img_arr = preprocess_input(img_arr)
    
    #Get Predicion probabilities
    preds_prob = normal_model.predict(np.array([img_arr]))
    
    #Prediction
    pred = np.argmax(preds_prob, axis = 1)
    
    #Ground Truth
    check = glob.glob('Unused unmasked/*0.jpg')
    if file in check:
        true = 1
    else:
        true = 0
        
    y_pred.append(pred[0])
    y_true.append(true)
#%%

normal_cm = metrics.confusion_matrix(y_true, y_pred)
accuracy = metrics.accuracy_score(y_true, y_pred)
balanced_accuracy = metrics.balanced_accuracy_score(y_true, y_pred)
precision = metrics.precision_score(y_true, y_pred)
recall = metrics.recall_score(y_true, y_pred)
f1 = metrics.f1_score(y_true, y_pred)

dictionary = {'Accuracy': [accuracy], 'Balanced Accuracy': [balanced_accuracy],
              'Precision': [precision], 
              'Recall': [recall], 'f1 Score': [f1]}
normal_metrics = pd.DataFrame(data = dictionary)





#%%

results = pd.concat([segmented_metrics, normal_metrics])
results.to_csv('comparison results.csv')

#%%

pd.DataFrame(segmented_cm).to_csv('Model confusion matrix.csv')
pd.DataFrame(normal_cm).to_csv('Comparison confusion matrics.csv')