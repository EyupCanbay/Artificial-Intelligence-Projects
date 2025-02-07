# WILL PEOPLE STOP USING THE BANK OR NOT / BANKAYI KULLLANMAYI BIRAKICAK MI BIRAKMICAK MI  

import pandas as pd
import numpy as nm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv("./datasets/Churn.csv")
print(df.head(4))

df = df.drop(["RowNumber","CustomerId", "Surname"] ,axis=1)
print(df.head(3))

ohe = OneHotEncoder()
xd = ohe.fit_transform(df[["Geography", "Gender"]]).toarray()
print(xd)

ohe.get_feature_names_out()
xd = pd.DataFrame(xd)

xd.columns=ohe.get_feature_names_out()
print(xd)

df=df.drop(columns=["Geography", "Gender"])

df[xd.columns]= xd
df.head(3)

y = df["Exited"]
x = df.drop(["Exited"], axis=1)

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=11,train_size=0.75)

rf = RandomForestClassifier()
model=rf.fit(x_train,y_train)
print(model.score(x_test,y_test))
