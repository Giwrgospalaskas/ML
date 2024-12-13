import pandas as pd
import math
from sklearn import linear_model
import numpy as np
from word2number import w2n


df = pd.read_csv('hiring.csv')
experience_col = df['experience']
new_experience_col = experience_col.apply(w2n.word_to_num)
df['experience'] = new_experience_col

df.fillna({'test_score(out of 10)' : math.floor(df['test_score(out of 10)'].mean())},inplace=True)

reg = linear_model.LinearRegression()
reg.fit(df.drop('salary($)', axis='columns'), df['salary($)'])

prediction1 = reg.predict([[2,9,6]])
prediction2 = reg.predict([[12,10,10]])
print(prediction1, prediction2)

