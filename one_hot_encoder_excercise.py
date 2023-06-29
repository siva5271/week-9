import pandas as pd
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression


df=pd.read_csv("carprices.csv")
le=LabelEncoder()
df.CarModel=le.fit_transform(df.CarModel)
X=df[["CarModel","Mileage","Age"]].values
y=df.SellPrice

ohe=OneHotEncoder(sparse=False)

ct=ColumnTransformer([("one_hot_encoder",ohe,[0])],remainder="passthrough")
x=ct.fit_transform(X)
x=x[:,1:]

model=LinearRegression()
model.fit(x,y)
print(model.predict([[0,1,86000,7]]))
print(model.score(x,y))