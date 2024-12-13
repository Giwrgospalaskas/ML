import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv('HR_comma_sep.csv')

subdf = df[['satisfaction_level', 'average_montly_hours', 'promotion_last_5years', 'salary']]
salary_dummies = pd.get_dummies(subdf['salary'], prefix="salary")

df_with_dummies = pd.concat([subdf,salary_dummies], axis='columns')

df_final = df_with_dummies.drop(['salary', 'salary_low'], axis='columns')

y = df['left']

X_train , X_test, Y_train, Y_test = train_test_split(df_final ,y, train_size= 0.3)

model = LogisticRegression()

model.fit(X_train, Y_train)

print(model.predict([[0.3, 700, 0, 0,0]]))

print(model.score(X_test, Y_test))
