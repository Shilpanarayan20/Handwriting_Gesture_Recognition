# -*- coding: utf-8 -*-
"""2DRESNET_KERAS_Main.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1pVaFKRujyHlWCvMz94-ip7VhohzKaMoo

Model
"""

import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, ReLU, GlobalAveragePooling2D, Add
from tensorflow.keras.layers import Conv1D, GlobalAveragePooling1D, BatchNormalization, Conv2D, MaxPool2D
from tensorflow import keras
import numpy as np

tf.config.experimental_run_functions_eagerly(True)

class ResnetBlock(tf.keras.Model):
    

    def __init__(self, channels: int, down_sample=False):
        super().__init__()

        self.__channels = channels
        self.__down_sample = down_sample
        self.__strides = [2, 1] if down_sample else [1, 1]

        KERNEL_SIZE = (3, 3)
        INIT_SCHEME = "he_normal"

        self.conv_1 = Conv2D(self.__channels, strides=self.__strides[0],
                             kernel_size=KERNEL_SIZE, padding="same", kernel_initializer=INIT_SCHEME)
        self.bn_1 = BatchNormalization()
        self.conv_2 = Conv2D(self.__channels, strides=self.__strides[1],
                             kernel_size=KERNEL_SIZE, padding="same", kernel_initializer=INIT_SCHEME)
        self.bn_2 = BatchNormalization()
        self.merge = Add()

        if self.__down_sample:
            self.res_conv = Conv2D(
                self.__channels, strides=2, kernel_size=(1, 1), kernel_initializer=INIT_SCHEME, padding="same")
            self.res_bn = BatchNormalization()

    def call(self, inputs):
        res = inputs

        x = self.conv_1(inputs)
        x = self.bn_1(x)
        x = tf.nn.relu(x)
        x = self.conv_2(x)
        x = self.bn_2(x)

        if self.__down_sample:
            res = self.res_conv(res)
            res = self.res_bn(res)

      
        x = self.merge([x, res])
        out = tf.nn.relu(x)
        return out


class ResNet18(tf.keras.Model):

    def __init__(self, num_classes, **kwargs):
        """
            num_classes: number of classes in specific classification task.
        """
        super().__init__(**kwargs)
        self.conv_1 = Conv2D(64, (7, 7), strides=2,
                             padding="same", kernel_initializer="he_normal")
        self.init_bn = BatchNormalization()
        self.pool_2 = MaxPool2D(pool_size=(2, 2), strides=2, padding="same")
        self.res_1_1 = ResnetBlock(64)
        self.res_1_2 = ResnetBlock(64)
        self.res_2_1 = ResnetBlock(128, down_sample=True)
        self.res_2_2 = ResnetBlock(128)
        self.res_3_1 = ResnetBlock(256, down_sample=True)
        self.res_3_2 = ResnetBlock(256)
        self.res_4_1 = ResnetBlock(512, down_sample=True)
        self.res_4_2 = ResnetBlock(512)
        self.avg_pool = GlobalAveragePooling2D()
        self.flat = Flatten()
        self.fc = Dense(num_classes, activation="softmax")

    def call(self, inputs):
        out = self.conv_1(inputs)
        out = self.init_bn(out)
        out = tf.nn.relu(out)
        out = self.pool_2(out)
        for res_block in [self.res_1_1, self.res_1_2, self.res_2_1, self.res_2_2, self.res_3_1, self.res_3_2, self.res_4_1, self.res_4_2]:
            out = res_block(out)
        out = self.avg_pool(out)
        out = self.flat(out)
        out = self.fc(out)
        return out

model = ResNet18(2)
model.build(input_shape = (None,2,98,1))
model.compile(optimizer = "adam",loss='binary_crossentropy', metrics=["accuracy"]) 
model.summary()

"""Data Loading"""

from sklearn.model_selection import train_test_split

X = np.load('/content/(N_W)X_Data.npy',allow_pickle=True)
Y = np.load('/content/(N_W)Y_Data.npy',allow_pickle=True)
X_train, X_test, y_train , y_test  = train_test_split(X, Y, test_size = 0.30, random_state = 150, shuffle=True)

X_train = np.asarray(X_train).astype('float32')
X_train = tf.reshape(tf.constant(X_train), [X_train.shape[0],2,98, 1])
y_train = tf.constant(y_train)
y_train = y_train 
y_train = tf.one_hot(y_train, depth = 2)
y_train = np.reshape(y_train,(8463,2))
y_train.shape

y_test = tf.constant(y_test)
y_test = y_test 
y_test = tf.one_hot(y_test, depth = 2)
y_test = np.reshape(y_test,(3627,2))
y_train.shape

y_test.shape

X_test.shape

"""Training"""

model = model.fit(X_train,y_train, epochs = 75,batch_size = 64)

"""Testing"""

#X_test = np.load('/content/X_test.npy',allow_pickle=True)
#Y_test = np.load('/content/Y_test.npy',allow_pickle=True)


X_test = np.asarray(X_test).astype('float32')
X_test = tf.reshape(tf.constant(X_test), [X_test.shape[0],2,98, 1])

model.evaluate(X_test,y_test)

from sklearn.metrics import accuracy_score

print("Accuracy Score = ", accuracy_score(YY_test, YY_predict))

model.save('/content/Binary_ResNet')



import torch

model = tf.keras.models.load_model('/content/sample_data')

"""Evaluation"""

Predict = model.predict(X_test)

YY = np.array(y_test)
YY_test = np.argmax(YY,axis = 1)

YY_predict = np.argmax(Predict,axis = 1)
YY_test = np.argmax(YY,axis = 1)

YY_predict = np.argmax(Predict,axis = 1)

YY_predict

from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
cm = confusion_matrix(YY_test, YY_predict)

cm

import pandas as pd


def print_confusion_matrix(confusion_matrix, class_names, figsize = (8,8),fontsize=14, normalize=True):
     
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

import matplotlib.pyplot as plt
import seaborn as sns

#class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A',
 #                  'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
  #                 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',
   #                'X', 'Y', 'Z']


#class_names = ['0,O,o','1,I,i,l ','2,Z,z','3','4','5,S,s','6,G','7','8','9,a,g,q','A',
 #                  'B', 'C,c', 'D,P,p,b', 'E,e', 'F,f', 'H,h', 'J,j ', 'K,k', 'L',
  #                 'M,m', 'N,n', 'Q','R','T,t', 'U,V,u,v', 'W,w','X,x', 'Y,y', 'd'] 

#class_names = ['0','1','2','3','4','5','6','7','8','9','A',
                # 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
                 # 'M', 'N', 'O', 'P', 'Q', 'R', 'S','T', 'U', 'V', 'W',
                 #'X', 'Y', 'Z','a',
                 #'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
                 # 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w',
                 # 'x', 'y', 'z'] 


class_names = ['Not Writing','Writing']                   
print_confusion_matrix(cm, class_names)
plt.savefig('Confusion_Matix_.png', dpi=300)

from sklearn.metrics import classification_report

#report = classification_report(Y_test, preds,target_names=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A',
 #                  'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
  #                 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',
   #                'X', 'Y', 'Z'])

#report = classification_report(YY_test, YY_predict,target_names=['0,O,o','1,I,i,l ','2,Z,z','3','4','5,S,s','6,G','7','8','9,a,g,q','A',
 #                  'B', 'C,c', 'D,P,p,b', 'E,e', 'F,f', 'H,h', 'J,j ', 'K,k', 'L',
  #                 'M,m', 'N,n', 'Q','R','T,t', 'U,V,u,v', 'W,w','X,x', 'Y,y', 'd']) 

#report = classification_report(YY_test, YY_predict,target_names= ['0','1','2','3','4','5','6','7','8','9','A',
 #                'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
  #               'M', 'N', 'O', 'P', 'Q', 'R', 'S','T', 'U', 'V', 'W',
   #               'X', 'Y', 'Z','a',
    #             'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
     #             'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w',
      #            'x', 'y', 'z']) 


report = classification_report(YY_test, YY_predict,target_names= ['Not Writing','Writing'])
print(report)

# Commented out IPython magic to ensure Python compatibility.
# %%capture cap --no-stderr
# print(report)

with open('Classification_Report.txt', 'w') as f:
    f.write(cap.stdout)

from collections import Counter


correct = [pred == true for pred, true in zip(YY_predict, YY_test)]
correct = np.array(correct).flatten()
print(Counter(correct))