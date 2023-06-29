import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

df=pd.read_csv("insurance_data.csv")
X_train, X_test,Y_train,Y_test = train_test_split(df[["age"]],df.bought_insurance)
model=LogisticRegression()
model.fit(X_train,Y_train)
print(model.predict([[55]]))
print(model.predict_proba([[55]]))