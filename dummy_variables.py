import pandas as pd
from sklearn.linear_model import LinearRegression

df=pd.read_csv("homeprices.csv")
#here i am getting the restructured format of all the categorical values
#each seperate caregory is represented in its own column with each one being represented by a boolean
dummies=pd.get_dummies(df.town)

#here the dummy variables is merged with the original table column wise
merged=pd.concat([df,dummies],axis='columns')

#after merging we delete the original column containing all the categorical values
#and also if we have n categorical values only n-1 columns are kept because the nth value can be predicted from the rest
merged=merged.drop(['monroe township',"town"],axis="columns")

model=LinearRegression()
X=merged.drop(["price"],axis="columns")
y=merged.price
model.fit(X,y)
print(model.score(X,y))