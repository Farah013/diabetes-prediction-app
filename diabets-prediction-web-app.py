# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 16:38:51 2022

@author: farah
"""

import numpy as np
import pickle 
import streamlit as slt

#Loading the saved model
loaded_model= pickle.load(open('C:/Users/farah/OneDrive/Bureau/ML/Projects/Diabets-prediciton/trained_model.sav', 'rb'))

#creating a function for prediction
def diabetes_prediction(input_data):
   
    #Changing the input_data to a numpy array
    input_data_as_np_array= np.asarray(input_data)

    #Reshape the array as we are prediciting for one instance
    input_data_reshaped= input_data_as_np_array.reshape(1, -1)

    prediction=loaded_model.predict(input_data_reshaped)
    print(prediction)

    if(prediction[0]==0):
      return "This person is not diabetic"
    else:
      return "This person is diabetic"


def main():
    
    #Giving a title to the user interface
    slt.title("Diabetes Prediction Web App")
    #Getting the input data from the user
    Pregnancies= slt.text_input('Number of Pregnancies')
    Glucose=slt.text_input('Glucose level')
    BloodPressure=slt.text_input('Blood pressure value')
    SkinThickness=slt.text_input('Skin Thickness value')
    Insulin=slt.text_input('Insulin level')
    BMI=slt.text_input('BMI value')
    DiabetesPadigreeFunction=slt.text_input('Diabetes Padigree Function value')
    Age=slt.text_input('Age of the person')
    
    #code for Predcition
    diagnosis=''
    #Creating a button for prediction
    if slt.button('Diabetes Test Result'):
        diagnosis=diabetes_prediction([Pregnancies, Glucose, BloodPressure,SkinThickness,Insulin,BMI,DiabetesPadigreeFunction, Age])
    slt.success(diagnosis)
    
if __name__ == '__main__':
    main()
        
    