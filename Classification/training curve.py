''' Code to extract training curves'''

#Import packages
import matplotlib.pyplot as plt
import pandas as pd

#%% load training log
df = pd.read_csv('comparison training.log')

#%% plot curves

#Accuracy Curves
fig1 = plt.gcf()
plt.plot(df['accuracy'])
plt.plot(df['val_accuracy'])
plt.axis(ymin=0.4,ymax=1)
plt.grid()
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['train', 'validation'])
plt.show()

#Loss Curves
plt.plot(df['loss'])
plt.plot(df['val_loss'])
plt.grid()
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['train', 'validation'])
plt.show()