# KALP KRİZİ GECİRME SINIFLANDIRMASI
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import xgboost as xgb
df = pd.read_csv("./datasets/heart.csv")
print(df.head(3))

y = df["target"]
x = df.drop(["target"],axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.75, random_state=20)
rf = xgb.XGBClassifier()
model = rf.fit(x_train,y_train)
print(model.score(x_test,y_test))



