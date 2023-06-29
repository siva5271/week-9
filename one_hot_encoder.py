from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
import pandas as pd

#first im using label encoder to convert the categorical data into numerical format
le=LabelEncoder()
df=pd.read_csv("homeprices.csv")
df.town=le.fit_transform(df.town)
X=df[["town","area"]].values
y=df.price

#next im using one hot encoder to convert that into an array with binary values
ohe=OneHotEncoder(sparse=False)
#here im specifying which column will have to be made into a categorical column
ct = ColumnTransformer(
    [('one_hot_encoder', ohe, [0])],
    remainder='passthrough'  
)

#here i am fitting the data into a model using dummy variables
x=ct.fit_transform(X)
x=x[:,1:]

model=LinearRegression()
model.fit(x, y)
print(model.predict([[0,1,3400]]))
# print(df)