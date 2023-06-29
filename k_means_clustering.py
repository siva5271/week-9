import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

df=pd.read_csv("income.csv")
#here i'm scaling the values to be between 0 and 1
scaler=MinMaxScaler(feature_range=(0,1))
scaler.fit(df[["Age"]])
df["Age"]=scaler.transform(df[["Age"]])
scaler.fit(df[["Income($)"]])
df["Income($)"]=scaler.transform(df[["Income($)"]])

#here i am using the elbow method and iterating over all the values
#of the possible number of centroids and finding their
#sum of square errors and i plotted a scatter plot to find
#the most suitable number using the elbow method
sse=[]
for i in range(1,len(df.Age)):
    kmean=KMeans(n_clusters=i)
    kmean.fit(df[["Age","Income($)"]])
    sse.append(kmean.inertia_)
# plt.xticks(np.arange(1,len(sse)+2,1))
# plt.scatter(range(1,len(sse)+1),sse)
# plt.show()

#i used the scatter plot and found out the most suitable 
#to be 3
kmean=KMeans(n_clusters=3)
y_predicted=kmean.fit_predict(df[["Age","Income($)"]])
df["cluster"]=y_predicted
#here i am diving the entire data frame into subsets acoording to the 
#to the different clusters and plotting them seperately
#along with their centroids
df0=df[df.cluster==0]
df1=df[df.cluster==1]
df2=df[df.cluster==2]
plt.scatter(df0["Age"], df0["Income($)"],color="red")
plt.scatter(df1["Age"], df1["Income($)"],color="blue")
plt.scatter(df2["Age"], df2["Income($)"],color="green")
plt.scatter(kmean.cluster_centers_[:,0],kmean.cluster_centers_[:,1],color="black")
plt.show()
# print(kmean.labels_)
