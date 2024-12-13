import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import seaborn as sn


flowers = load_iris()

X_train, X_test, Y_train, y_test = train_test_split(flowers.data, flowers.target, test_size=0.1)

model = LogisticRegression()
model.fit(X_train,Y_train)

print(len(X_test))

y_predicted = model.predict(X_test)



cm = confusion_matrix(y_test, y_predicted)
print(model.score(X_test, y_test))

print(cm)

plt.figure(figsize=(10,7))
sn.heatmap(cm, annot= True)
plt.xlabel('Predicted')
plt.ylabel('Truth')

plt.show()