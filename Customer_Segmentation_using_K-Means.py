#---------------------------------------ABOUT DATASET
#Content
#You are owing a supermarket mall and through membership cards , you have some basic data about your customers like Customer ID, age, gender, annual income and spending score.
#Spending Score is something you assign to the customer based on your defined parameters like customer behavior and purchasing data.
#Problem Statement
#You own the mall and want to understand the customers like who can be easily converge [Target Customers] so that the sense can be given to marketing team and plan the strategy accordingly.

#------------------------------------------WORK FLOW
#data collection
#data analysis 
#data separation
#data train test split
#data model use 
#data prediction 
#------------------------------------------IMPORT LABRARY
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt

#--------------------------------------------DATA ANLYSIS

data = pd.read_csv("C:/Users/kunde/all vs code/ml prject/Mall_Customers.csv")
print(data.shape)
print(data.columns)
print(data.head(5))
print(data.tail(5))
print(data.isnull().sum())
print(data.info())
print(data.describe())
data.replace({"Gender":{"Male":1, "Female":0}}, inplace=True)
print(data.head(5))

#----------------------------------------------data separte 
x = data.drop(columns=["CustomerID", "Spending Score (1-100)"])
print(x.head(5))
y = data["Spending Score (1-100)"]

#-----------------------------------------train test split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)
print(x.shape, x_test.shape, x_train.shape)

#-----------------------------------------model selection
model = RandomForestRegressor()
model.fit(x_train, y_train)

#----------------------------------------prediction of train data
y_train_p = model.predict(x_train)
print(y_train_p, "this is our prediction", np.array(y_train), "this is true prediction")
accur = metrics.r2_score(y_train_p, y_train)
print(accur, "this score of train data")

#---------------------------------------prediction of test data
y_test_p = model.predict(x_test)
print(y_test_p,"this is our data prediction", np.array(y_test), "this is true data")
accur = metrics.r2_score(y_test, y_test_p)
print(accur, "this score of test data")

#---------------------------------------prediction of single data
x = [1, 21, 15]#81
x_new = np.asarray(x)
x_reshape = x_new.reshape(1, -1)
y_pred = model.predict(x_reshape)
print(y_pred)





from xgboost import XGBRegressor
#-================================================
model= XGBRegressor()
model.fit(x_train, y_train)

#----------------------------------------prediction of train data
y_train_p = model.predict(x_train)
print(y_train_p, "this is our prediction", np.array(y_train), "this is true prediction")
accur = metrics.r2_score(y_train_p, y_train)
print(accur, "this score of train data")

#---------------------------------------prediction of test data
y_test_p = model.predict(x_test)
print(y_test_p,"this is our data prediction", np.array(y_test), "this is true data")
accur = metrics.r2_score(y_test, y_test_p)
print(accur, "this score of test data")

#---------------------------------------prediction of single data
x = [1, 21, 15]#81
x_new = np.asarray(x)
x_reshape = x_new.reshape(1, -1)
y_pred = model.predict(x_reshape)
print(y_pred)


















