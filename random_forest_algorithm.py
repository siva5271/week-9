import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn

digits=load_digits()
df=pd.DataFrame(digits.data)
df["target"]=digits.target
X=df.drop(["target"],axis='columns')
y=df.target
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
model=RandomForestClassifier(n_estimators=100)
model.fit(X_train,y_train)
y_predicted=model.predict(X_test)
cm=confusion_matrix(y_test,y_predicted)
plt.figure(figsize=(8,6))
sn.heatmap(cm,annot=True)
plt.xlabel("True")
plt.ylabel("Predicted")
plt.show()