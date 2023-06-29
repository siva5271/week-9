import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

df=pd.read_csv("titanic.csv")
inputs=df.drop(["PassengerId","Name","SibSp","Parch","Ticket","Cabin","Embarked","Survived"],axis="columns")
target=df.Survived
le=LabelEncoder()
labels=le.fit_transform(inputs["Sex"])
inputs.drop(["Sex"],axis="columns")
inputs["Sex"]=labels
inputs["Age"]=inputs["Age"].fillna(int(inputs["Age"].mean()))
X_train, X_test,y_train,y_test = train_test_split(inputs,target,test_size=0.1)
nb=GaussianNB()
nb.fit(X_train,y_train)
print(nb.score(X_test,y_test))