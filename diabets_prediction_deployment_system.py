# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 15:49:28 2022

@author: farah
"""

import numpy as np 
import pickle

#Loading the saved model
loaded_model= pickle.load(open('C:/Users/farah/OneDrive/Bureau/ML/Projects/Diabets-prediciton/trained_model.sav', 'rb'))

#Predicting an input
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