import pandas as pd
from sklearn import linear_model

df = pd.read_csv('carprices.csv')
dummies = pd.get_dummies(df['Car Model'])
merged = pd.concat([df,dummies], axis='columns')
final = merged.drop(['Car Model', 'Mercedez Benz C class'], axis='columns')
print(final)

model = linear_model.LinearRegression()
x = final.drop('Sell Price($)', axis='columns')
y = final['Sell Price($)']
print(x,y)
model.fit(x,y)
print(model.predict([[45000, 4, 0,0]]))
print(model.predict([[86000, 7, 0,1]]))
print(model.score(x,y))