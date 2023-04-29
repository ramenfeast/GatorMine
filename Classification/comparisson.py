'''Comparison Code'''

''' Classifying model trainer'''

#Import packages
import tensorflow as tf
from tensorflow.python.keras.layers import Dense, Flatten, Lambda
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.resnet50 import preprocess_input, ResNet50

#%% Load training dataset with 70-30 split

img_height,img_width=180,180
batch_size=2
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  'ForComparison',
  validation_split=0.3,
  subset="training",
  seed=0,
  image_size=(img_height, img_width),
  batch_size=batch_size)

#%% Load validation dataset with 70-30 split
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  'ForComparison',
  validation_split=0.3,
  subset="validation",
  seed=0,
  image_size=(img_height, img_width),
  batch_size=batch_size)

#%% Print class names: [0 No Diabetes, 1 Diabetes]
class_names = train_ds.class_names
print(class_names)
#%% Create the Model
model = Sequential()

#Download ResNet50
pretrained = ResNet50(
    include_top=False,
    weights="imagenet",
    input_shape=(180,180,3),
    pooling= 'avg',
    classes=2
)

#Freeze layer weights in Resnet 50
for layer in pretrained.layers:
        layer.trainable=False

#Model Design: Preprocessing layer -> ResNet50 -> Flatten ->
#   -> Dense Layer w ReLU -> Dense Layer w SoftMax
model.add(Lambda(lambda x: preprocess_input(x)))        
model.add(pretrained)
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(2, activation='softmax'))

#%% Model Compiler
# LR = 0.01, Optimizer = Adam, loss = sparse catergorical crossentropy b/c two classes
model.compile(optimizer=Adam(learning_rate=0.01),loss='sparse_categorical_crossentropy',metrics=['accuracy'])

#%% Call Back Creation

#Scheduler: Reduces LR by 0.1 every 10 epochs
def scheduler(epoch, learning_rate):
    if epoch%10 != 0:
        return learning_rate
    else:
        return learning_rate*0.1
    
lr_schedule = tf.keras.callbacks.LearningRateScheduler(scheduler)

#Logs training and validation accuracy and loss
csv_logger = tf.keras.callbacks.CSVLogger('comparison training.log')

#Early stopper, stops training when loss no longer decreases for 3 epochs
early_stop = tf.keras.callbacks.EarlyStopping(monitor = 'loss', patience = 3)
#%% Model Training

epochs=100 #Epochs set to 100 but early stopper will stop at around 65

history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs,
  callbacks = [csv_logger, early_stop, lr_schedule]
)
#%%

#Save model to make loadable
model.save('comparison_model')

#%% Model Summary
model.summary()
