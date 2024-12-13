from sklearn import linear_model
import pickle

with open('model_pickle', 'rb') as f:
    mp = pickle.load(f)

print(mp.coef_)