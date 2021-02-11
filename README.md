# Prediction of Area burnt by Forest Fires

In the present times, one of the serious threats to the environment is climate change. A direct result of climate change is Forest fires which has the capability to cause severe damage to the ecology. This project is an attempt to predict the area burnt by a potential forest fire. The problem is formulated as a regression problem based on meteorological factors that influences forest fires such as humidity, wind speed, rain etc., available in the Forest Fires Dataset.

## Dataset

### Overview
The Forest Fires dataset is available in the UCI Machine Learning Repository. It consists of details regarding forest fires in northeast region of Portugal.

### Task
The aim is to predict the burnt area. The features are:
 
 1. X - x-axis spatial coordinate within the Montesinho park map: 1 to 9
 1. Y - y-axis spatial coordinate within the Montesinho park map: 2 to 9
 1. month - month of the year: 'jan' to 'dec'
 1. day - day of the week: 'mon' to 'sun'
 1. FFMC - FFMC index from the FWI system: 18.7 to 96.20
 1. DMC - DMC index from the FWI system: 1.1 to 291.3
 1. DC - DC index from the FWI system: 7.9 to 860.6
 1. ISI - ISI index from the FWI system: 0.0 to 56.10
 1. temp - temperature in Celsius degrees: 2.2 to 33.30
 1. RH - relative humidity in %: 15.0 to 100
 1. wind - wind speed in km/h: 0.40 to 9.40
 1. rain - outside rain in mm/m2 : 0.0 to 6.4
 1. area - the burned area of the forest (in ha): 0.00 to 1090.84

### Access
* The `TabularDatasetFactory` class contains a method called `from_delimited_files` which takes a url as input and creates a `TabularDataset`. This dataset can be then converted to a DataFrame using `to_pandas_dataframe` method.
* dataset url: http://archive.ics.uci.edu/ml/machine-learning-databases/forest-fires/forestfires.csv

## Automated ML
Settings and Configurations of AutoML:

* Primary metric: Normalized Mean Absolute Error
  It gives a normalized score of the average difference between the predicted and actual burnt area.
  
* Cross Validation: 5
  The dataset will be split into 5 parts. Four of them will be used for training and one for testing. This is repeated until each part has become the test set once.
  
* Experiment Timeout Minutes: 30

* Max Concurrent Iterations: 4 ( Same as maximum number nodes in compute cluster )

### Results
* The best AutoML model is MaxAbsScaler,Gradient boosting with a Mean Absolute Error of 12.989 . The error can be further reduced by running the experiment longer and enabling Deep Learning.

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search


### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?
 b
*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

## Screen Recording
* https://youtu.be/S_pfUnAwa64
