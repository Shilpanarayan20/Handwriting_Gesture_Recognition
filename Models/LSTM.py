#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Conv1D
from keras.layers import Input
from keras.layers import MaxPooling1D
from keras import Model
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import tensorflow as tf
import seaborn as sns


# Data_Loading

# In[11]:




#X = np.load("Prepared/Not_Writng_Not/Feature_N_W.npy",allow_pickle=True)
#Y = np.load("Prepared/Not_Writng_Not/Feature_N_W_Y.npy",allow_pickle=True)

#X_train, X_test, y_train , y_test  = train_test_split(X, Y, test_size = 0.30, random_state = 150, shuffle=True)

X_train = np.load("X_Train.npy",allow_pickle=True)
X_test = np.load("X_Test.npy",allow_pickle=True)
y_train = np.load("Y_Train.npy",allow_pickle=True)
y_test = np.load("Y_Test.npy",allow_pickle=True)

X_train = np.asarray(X_train).astype('float32')
X_test = np.asarray(X_test).astype('float32')


# In[19]:


#y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

#y_train = np.asarray(y_train).astype('float32')
y_test = np.asarray(y_test).astype('float32')


# In[20]:


y_test.shape


# Model

# In[13]:


model = Sequential()

model.add(LSTM(100,return_sequences=True, input_shape=(2,98)))
model.add(LSTM(100,return_sequences=True))
model.add(LSTM(150,return_sequences=True))
model.add(LSTM(200))
model.add(Dropout(0.3))
model.add(Dense(units = 2, activation='softmax'))


# In[14]:


model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=['accuracy'])


# Training

# In[15]:


history = model.fit(X_train, y_train,epochs=75, batch_size=64)


# Testing

# In[21]:


model.evaluate(X_test,y_test)


# In[22]:


preds = model.predict(X_test).argmax(1)


# In[23]:


Y_test = y_test.argmax(1)   


# In[24]:


from sklearn.metrics import accuracy_score

print("Accuracy Score = ", accuracy_score(Y_test, preds))


# Evaluation

# In[25]:


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


# In[26]:


from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
cm = confusion_matrix(Y_test, preds)

cm


# In[27]:


import seaborn as sns

class_names = ['Not Writing','Writing']
print_confusion_matrix(cm, class_names)
plt.savefig('Confusion_Matrix_LSTM(x,y).png', dpi=300)


# In[28]:


report = classification_report(Y_test, preds,target_names=['Not_Writng', 'Writng'])
print(report)


# In[29]:


from collections import Counter


correct = [pred == true for pred, true in zip(preds, Y_test)]
correct = np.array(correct).flatten()
print(Counter(correct))


# In[39]:


YY = np.array(Y_test).flatten()
classifiedIndexes = np.where(YY==preds)[0]
misclassifiedIndexes = np.where(YY!=preds)[0]


# In[33]:


A = 2343
P = preds[A]
Y = Y_test[A]

Y


# In[34]:


P


# In[37]:


x_test = np.reshape(X_test,(3627,2,98))
X = x_test
XX = X[A]


# In[38]:


import matplotlib.pyplot as plt

x = XX[0]
y = XX[1]

plt.plot(x, y)
plt.title('X Y Co-ordinates')
plt.savefig("MC6.png")

plt.show()


# In[ ]:





# In[ ]:




