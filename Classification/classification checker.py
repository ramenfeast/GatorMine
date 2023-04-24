
''' Code to Check the model performance on unused retinal images'''

import numpy as np
from keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
import tensorflow.keras.utils as image

#%% Loading Saved Model

model = load_model('keras_model')

#%% Loading Test Image
img = image.load_img('masked_images/test/17_test_0.jpg')

# Resize the image
img = img.resize((180, 180))

# Convert the image to a NumPy array
img_arr = np.array(img)

# Preprocess the image
img_arr = preprocess_input(img_arr)

#%%
# Feed the image to the model and get the predictions
preds_prob = model.predict(np.array([img_arr]))
print(preds_prob)
print(np.argmax(preds_prob, axis=1))