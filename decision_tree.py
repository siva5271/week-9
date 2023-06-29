import pandas as pd 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import tree

df=pd.read_csv("salaries.csv")
input=df.drop("salary_more_then_100k",axis="columns")
result=df.salary_more_then_100k
le=LabelEncoder()
for column in input:
    input[column]=le.fit_transform(input[column])
X_train,X_test,Y_train,Y_test=train_test_split(input,result,test_size=0.1)
model=tree.DecisionTreeClassifier()
model.fit(X_train,Y_train)
print(model.predict([[1,2,0]]))