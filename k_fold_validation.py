import pandas as pd
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold,cross_val_score

digits=load_digits()
df=pd.DataFrame(digits.data)
df["target"]=digits.target
X=df.drop(["target"],axis="columns")
y=df.target
svf=StratifiedKFold(n_splits=10)
scores_linear=cross_val_score(LinearRegression(),X,y,cv=svf)
scores_logistic=cross_val_score(LogisticRegression(),X,y,cv=svf)
scores_rfc=cross_val_score(RandomForestClassifier(),X,y,cv=svf)
scores_svm=cross_val_score(SVC(),X,y,cv=svf)
print(scores_linear.mean())
print(scores_logistic.mean())
print(scores_rfc.mean())
print(scores_svm.mean())
