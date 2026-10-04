import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import kagglehub
import warnings

from kagglehub import KaggleDatasetAdapter
from sklearn.linear_model import LinearRegression  #what exactly you want from the Library(select model)
from sklearn.impute import KNNImputer  #what exactly you want from the Library()
from sklearn.model_selection import train_test_split #spliting data for training
from sklearn.preprocessing import StandardScaler #standerdises data after spliting

######################################################importing Data set
data = pd.read_csv("Life Expectancy Data.csv")
######################################################

print(data.head(22)) #rows and columns to print
print(data.columns) #shows columns
print(data.describe()) #describe your data statistics
print(data.tail()) #prints last few rows
print(data.shape) #number of rows and columns
print(data.info()) #info of your data

######################################################Cleaning
missing = data.isnull().sum()
print(missing)

duplicates = data.duplicated().sum()
print("duplicate values:")
print(duplicates)

for i in data.select_dtypes(include="object").columns:
    print(data[i].value_counts())
    print("**************" * 8)