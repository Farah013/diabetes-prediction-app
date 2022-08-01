# -*- coding: utf-8 -*-
"""Diabets-predicition-Deployment.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1hxDcSCUNJiD_DuhWmStCFW7Qbpi1QFct

# Importing the Dependencies
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import svm

"""# Data Collection and Analysis"""

#Loading the diabetes dataset to a pandas DataFrame
dataset=pd.read_csv("diabetes.csv")

#Printing the first 5 rows of the dataset
dataset.head()

# Number of rows and columns in this dataset
dataset.shape

# Getting the statistical measures of the data
dataset.describe()

dataset['Outcome'].value_counts()

"""0--> Non-Diabetic
1-->Diabetic
"""

dataset.groupby('Outcome').mean()

#Seperating the data and labels
X= dataset.drop(columns='Outcome', axis=1)
Y= dataset['Outcome']

print(X)

print(Y)

"""# Train Test Split"""

X_train, X_test, Y_train, Y_test= train_test_split(X,Y, test_size=0.2, stratify=Y,random_state=2)

print(X.shape, X_train.shape, X_test.shape)

"""# **Training the model**"""

classifier=svm.SVC(kernel='linear')

#Training the support vector machine classifier
classifier.fit(X_train,Y_train)

"""# Model Evaluation

Accuracy score
"""

# Accuracy score on the training data
X_train_prediction= classifier.predict(X_train)
training_data_accuracy= accuracy_score(X_train_prediction, Y_train)

print('Accuracy score of the training data ', training_data_accuracy)

# Accuracy score on the test data
X_test_prediction= classifier.predict(X_test)
test_data_accuracy= accuracy_score(X_test_prediction, Y_test)

print('Accuracy score of the test data ',test_data_accuracy)

"""# Making a Predictive System"""

input_data=(4,110,92,0,0, 37.6, 0.191,30)
#Chnaging the input_data to a numpy array
input_data_as_np_array= np.asarray(input_data)

#Reshape the array as we are prediciting for one instance
input_data_reshaped= input_data_as_np_array.reshape(1, -1)

prediction=classifier.predict(input_data_reshaped)
print(prediction)

if(prediction[0]==0):
  print("This person is not diabetic")
else:
  print("This person is diabetic")

"""# Saving the trained Model"""

import pickle

filename= 'trained_model.sav'
pickle.dump(classifier, open(filename, 'wb'))

#Loading the saved model
loaded_model= pickle.load(open('trained_model.sav', 'rb'))

input_data=(4,110,92,0,0, 37.6, 0.191,30)
#Chnaging the input_data to a numpy array
input_data_as_np_array= np.asarray(input_data)

#Reshape the array as we are prediciting for one instance
input_data_reshaped= input_data_as_np_array.reshape(1, -1)

prediction=loaded_model.predict(input_data_reshaped)
print(prediction)

if(prediction[0]==0):
  print("This person is not diabetic")
else:
  print("This person is diabetic")