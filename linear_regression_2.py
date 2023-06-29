import numpy as np
import pandas as pd
from sklearn import linear_model
import math as math

df=pd.read_csv("homeprices.csv")
# here i am doing a bit of pre processing where in i'm replacing all the values with the median
replacement=math.floor(df.bedrooms.median())
df.bedrooms=df.bedrooms.fillna(replacement)

#the model is built in the following line
reg=linear_model.LinearRegression()
reg.fit(df[["area","bedrooms","age"]],df.price)

print(reg.predict([[2500,4,5]]))