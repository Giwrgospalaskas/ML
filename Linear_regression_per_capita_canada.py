import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model 
import pickle
import joblib

df = pd.read_csv('canada_per_capita_income.csv')
new_df = df.drop('per capita income (US$)', axis= 'columns')
income = df['per capita income (US$)']
reg = linear_model.LinearRegression()
reg.fit(new_df, income)

predict = reg.predict([[2020]])
print(predict)

with open('model_pickle', 'wb') as f:
    pickle.dump(reg, f)

joblib.dump(reg, 'model_joblib')
