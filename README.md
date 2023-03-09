# Handwriting Gesture Recognition

1) Implementation of segmenting writing activity from random gestures or activities performed in air using finger worn sensor.
 
2) Implementation of prediction of characters and numbers.

sample examples for both tasks:


* Not Writing and Writing activity segmentation:


https://user-images.githubusercontent.com/70266536/223907702-7f1f0b04-bf53-4819-a8db-7ab0e1aed2aa.mp4



* Characters and numbers segmentation:


 
https://user-images.githubusercontent.com/70266536/223907768-4c8a08aa-a175-4c2f-a989-3096c43af58a.mp4



# Project files description

Consist of three folders :

1) Dataset which consist of prepared data for all three data representation for both tasks 
 
2) Data preprocessing file where we have code for Enlarge (upsampling), Reduce (Downsampling) and Padding

3) Models - 5 different segmentations networks .ipynb files and .py files : Bi-LSTM , LSTM , ResNet , OS-CNN , GTN 

4) Best Models - Saved models of Best performed network for each data representation and for each category of classes (Writing).

# Environment

* python == 3.8

* pytorch == 1.13.1

* Keras = 2.11.0

* scikit-learn == 1.2.1

# How to run network

* Try Google Colab

* Import the network .ipynb file from model\ipynb_files folder

  or

* Run With Jupyter Notebook

* Can be trained and tested with the data available in dataset folder or can replace the X_train, Y_train, X_test, Y_test as you want.

# How to predict using saved models

This code could help you to load model and use the model for prediction
can be loaded using 

* model = torch.load('path/to/location' or 'filename') -> (for .pt file) -> (it needs model trained by Netwrok.py)
 
* model= keras.models.load_model('path/to/location' or 'filename') -> (for .h5 file)
 
* model = keras.models.load_model('path/to/location' or 'filename') -> (load complete zip folder)

then evaluate using the test dataset 


