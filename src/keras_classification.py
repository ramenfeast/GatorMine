# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 15:02:26 2023

@author: camer
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

#%%

img_height,img_width=180,180
batch_size=2
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  'ForKerasClassification',
  validation_split=0.3,
  subset="training",
  seed=0,
  image_size=(img_height, img_width),
  batch_size=batch_size)

#%%
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  'ForKerasClassification',
  validation_split=0.3,
  subset="validation",
  seed=0,
  image_size=(img_height, img_width),
  batch_size=batch_size)


#%%
class_names = train_ds.class_names
print(class_names)

#%%
model = Sequential()

pretrained = tf.keras.applications.ResNet50V2(
    include_top=False,
    weights="imagenet",
    input_shape=(180,180,3),
    pooling= 'avg',
    classes=2
)

for layer in pretrained.layers:
        layer.trainable=False
        
model.add(pretrained)
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(2, activation='softmax'))

#%%
model.summary()

#%%

model.compile(optimizer=Adam(learning_rate=0.001),loss='sparse_categorical_crossentropy',metrics=['accuracy'])

#%%
csv_logger = tf.keras.callbacks.CSVLogger('training.log')
early_stop = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 6)
#%%
epochs=500

history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs,
  callbacks = [csv_logger, early_stop]
)



model.save('keras_model')



#%% DO NOT RUN

fig1 = plt.gcf()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.axis(ymin=0.4,ymax=1)
plt.grid()
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['train', 'validation'])
plt.show()
     
#%% DO NOT RUN

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.grid()
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['train', 'validation'])
plt.show()
#%%
import cv2
image=cv2.imread("ForKerasClassification\1 diabetes\26_manual1.jpg")
image = cv2.resize(image, (180,180))
cv2.imshow('window',image)