import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier

iris=load_iris()
df=pd.DataFrame(iris.data,columns=iris.feature_names)
df["target"]=iris.target
df["flower_name"]=df.target.apply(lambda x: iris.target_names[x])
X=df.drop(["target","flower_name"],axis="columns")
y=df["target"]
X_train,X_test,y_train,y_test = train_test_split(X,y)
model=KNeighborsClassifier(n_jobs=2)
model.fit(X_train,y_train)
print(model.score(X_test,y_test))

