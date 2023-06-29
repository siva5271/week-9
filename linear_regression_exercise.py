import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

df=pd.read_csv("canada_per_capita_income.csv")
reg=linear_model.LinearRegression()
reg.fit(df[["year"]], df["per capita income (US$)"])
print(reg.predict([[2020]]))
plt.scatter(df[["year"]], df["per capita income (US$)"])
plt.plot(df.year, reg.predict(df[["year"]]))
plt.show()