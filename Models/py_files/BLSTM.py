#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Bidirectional
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import seaborn as sns
from keras import utils as np_utils
import tensorflow as tf
from tensorflow import keras


# In[2]:


pip install numpy==1.19.5


# Data_Loading

# In[6]:



X = np.load("Prepared/(N_W)X_Data.npy",allow_pickle=True)
Y = np.load("Prepared/(N_W)Y_Data.npy",allow_pickle=True)

#y_train = np.load("Prepared/Not_Writng_Not/N_W_Y_train.npy")
#y_test = np.load("Prepared/Not_Writng_Not/N_W_Y_test.npy")

#X_train = np.load("Prepared/Not_Writng_Not/X&Y_Co-ordinates/N_W_X_train.npy",allow_pickle=True)
#X_test = np.load("Prepared/Not_Writng_Not/X&Y_Co-ordinates/N_W_X_test.npy",allow_pickle=True)
#y_train = np.load("Prepared/Not_Writng_Not/X&Y_Co-ordinates/N_W_Y_train.npy",allow_pickle=True)
#y_test = np.load("Prepared/Not_Writng_Not/X&Y_Co-ordinates/N_W_Y_test.npy",allow_pickle=True)

X_train, X_test, y_train , y_test  = train_test_split(X, Y, test_size = 0.30, random_state = 150, shuffle=True)


X_train = np.asarray(X_train).astype('float32')
X_test = np.asarray(X_test).astype('float32')

y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

y_train = np.asarray(y_train).astype('float32')
y_test = np.asarray(y_test).astype('float32')
             


# In[8]:


X_test.shape


# Model

# In[23]:


model = Sequential()


# In[24]:


model = Sequential()
model.add(Bidirectional(LSTM(2,return_sequences=True, input_shape=(2,98))))
model.add(Bidirectional(LSTM(4,return_sequences=True)))
model.add(Bidirectional(LSTM(8,return_sequences=True)))
model.add(Bidirectional(LSTM(16)))
model.add(Dropout(0.3))
model.add(Dense(units = 2, activation='softmax'))


# In[25]:


model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=['accuracy'])


# Training

# In[26]:


history = model.fit(X_train, y_train,epochs=75, batch_size=64)


# In[27]:


model.summary()


# Testing

# In[28]:


# fit the model
preds = model.predict(X_test)


# In[29]:


predict = np.argmax(preds,axis = 1)
predict


# In[30]:


model.evaluate(X_test,y_test)


# In[31]:


y_train = np.array(y_train)


# In[32]:


y_test.shape


# In[33]:


y_test = np.argmax(y_test,axis = 1)
y_test


# In[41]:


from sklearn.metrics import accuracy_score

print("Accuracy Score = ", accuracy_score(y_test, predict))


# Evaluation

# In[42]:


from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
cm = confusion_matrix(y_test, predict)

cm


# In[48]:


def print_confusion_matrix(confusion_matrix, class_names, figsize = (8,8),
                           fontsize=14, normalize=True):
     
    if normalize:
        confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'

    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names,
    )
    fig = plt.figure(figsize=figsize)
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt= fmt)
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# In[51]:


import seaborn as sns

class_names = ['Not Writing','Writing']
print_confusion_matrix(cm, class_names)
plt.savefig('Confusion_Matrix_BLSTM(x,y).png', dpi=300)


# In[52]:


report = classification_report(y_test, predict,target_names=['Not_Writng', 'Writng'])
print(report)


# In[53]:


from collections import Counter


correct = [pred == true for pred, true in zip(predict, y_test)]
correct = np.array(correct).flatten()
print(Counter(correct))


# In[54]:


YY = np.array(y_test).flatten()
classifiedIndexes = np.where(YY==predict)[0]
misclassifiedIndexes = np.where(YY!=predict)[0]


# In[ ]:





# In[79]:


A = 789
P = predict[A]
Y = y_test[A]

Y


# In[80]:


P


# In[81]:


X_test = np.asarray(X_test).astype('float32')
y_test = np.asarray(y_test).astype('float32')
X_test = np.reshape(X_test,(3627,2,98))


# In[82]:


y_test.shape


# In[83]:


X = X_test[A]
XX = X


# In[84]:


y_test[A]


# In[85]:


import matplotlib.pyplot as plt

x = XX[0]
y = XX[1]

plt.plot(x, y)
plt.title('X Y Co-ordinates')
plt.savefig("0_MC2_1.png")

plt.show()

