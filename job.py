import joblib

md = joblib.load('model_joblib')

print(md.coef_)
