import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

#here we are reading the data from a file w
df=pd.read_csv("homeprices.csv")
reg=linear_model.LinearRegression()

#the reg obj is trained with the model in the next step
reg.fit(df[["area"]], df["price"])

df2=pd.read_csv("areas.csv")
#the areas file  only contains the areas so we created an array of values using
#the model we trained from homeprices file
predicted_values=reg.predict(df2[["area"]])

#in the next step we are adding the new values into the original areas file
df2["price"]=predicted_values
df2.to_csv("areas.csv",index=False)

plt.xlabel("Sq. feet")
plt.ylabel("price")
#now we are plotting the points on a scatterplot
plt.scatter(df.area,df.price,color="red")
plt.scatter(df2.area, df2.price,color="blue")
plt.plot(df.area,reg.predict(df[["area"]]),"")
plt.plot(df2.area,reg.predict(df2[["area"]]),color="green")
plt.show()