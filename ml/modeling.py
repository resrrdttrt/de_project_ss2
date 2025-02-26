# required libraries
import numpy as np # data conversion
import pandas as pd # data processing, CSV file I/O
import os # accessing directory structure

# Monitoring input files
print(os.listdir('../dataset'))

data_Item_Store_Sale = pd.read_csv('../dataset/Item_Store_Sales.csv')

data_Stores = pd.read_csv('../dataset/Stores.csv')

data_Items = pd.read_csv('../dataset/Items.csv')

data_frame = pd.merge(data_Item_Store_Sale, data_Items, on='Item_Id')
data_frame = pd.merge(data_frame, data_Stores, on='Store_Id')

# Data Preprocessing
## Data Manipulation
data_frame.isnull().sum()
data_frame['Item_Fabric_Amount'] = data_frame['Item_Fabric_Amount'].fillna(data_frame['Item_Fabric_Amount'].mean())
mode_of_store_size = data_frame.pivot_table(values='Store_Size', columns='Store_Type', aggfunc=lambda x: x.mode()[0])
miss_values = data_frame['Store_Size'].isnull()
data_frame.loc[miss_values, 'Store_Size'] = data_frame.loc[miss_values, 'Store_Type'].apply(lambda x: mode_of_store_size[x])

## Data Cleaning
data_frame.duplicated().sum()

## Data Conversion
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
data_frame['Store_Id'] = encoder.fit_transform(data_frame['Store_Id'])
data_frame['Store_Size'] = encoder.fit_transform(data_frame['Store_Size'])
data_frame['Store_Location_Type'] = encoder.fit_transform(data_frame['Store_Location_Type'])
data_frame['Store_Type'] = encoder.fit_transform(data_frame['Store_Type'])

data_frame['Item_Id'] = encoder.fit_transform(data_frame['Item_Id'])
data_frame['Item_Fit_Type'] = encoder.fit_transform(data_frame['Item_Fit_Type'])
data_frame['Item_Fabric'] = encoder.fit_transform(data_frame['Item_Fabric'])

## Data Splitting (Test and Train)
from sklearn.model_selection import train_test_split
X = data_frame.drop(columns='Item_Store_Sales', axis=1)
Y = data_frame['Item_Store_Sales']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# Machine Learning Model Training
from sklearn import metrics # for modelling evaluation
import pickle # for saving and loading models

## Random Forest
from sklearn.ensemble import RandomForestRegressor
regressor_randomForest = RandomForestRegressor()
regressor_randomForest.fit(X_train, Y_train)
y_pred_randomForest = regressor_randomForest.predict(X_test)
# Model Evaluation
print("R-squared Score:", metrics.r2_score(Y_test, y_pred_randomForest))
# Saving trained model
pickle.dump(regressor_randomForest, open('local_model/model_randomForest.pkl','wb'))
print("Saved model")


## XGBoost
from xgboost import XGBRegressor
regressor_xgboost = XGBRegressor()
regressor_xgboost.fit(X_train, Y_train)
y_pred_xgb = regressor_xgboost.predict(X_test)
# Model Evaluation : Metrics for Continuous Target Regression
print("R-squared Score:", metrics.r2_score(Y_test, y_pred_xgb))
# Saving trained model
pickle.dump(regressor_xgboost, open('local_model/model_xgboost.pkl','wb'))
print("Saved model")

## Decision Tree
from sklearn.tree import DecisionTreeRegressor
regressor_decisionTree = DecisionTreeRegressor()
regressor_decisionTree.fit(X_train, Y_train)
y_pred_decisionTree = regressor_decisionTree.predict(X_test)
# Model Evaluation
print("R-squared Score:", metrics.r2_score(Y_test, y_pred_decisionTree))
# Saving trained model
pickle.dump(regressor_decisionTree, open('local_model/model_decisionTree.pkl','wb'))
print("Saved model")


