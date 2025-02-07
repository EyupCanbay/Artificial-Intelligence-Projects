#AYNIK NE KADAR HARCAMA YAPILLACAK TAHMİNİ
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

df= pd.read_csv("./datasets/insurance.csv")

df = pd.get_dummies(df, columns=["sex", "smoker", "region"],drop_first=True, dtype=int)
print(df.head(3))

y = df['charges']
x = df.drop(["charges"], axis=1)

x_train, x_text, y_train, y_text = train_test_split(x,y,test_size=0.8, random_state=5)
lr = LinearRegression()
model = lr.fit(x_train, y_train)
print(model.score(x_text,y_text))


rf = RandomForestRegressor(n_estimators=200)
model = rf.fit(x_train, y_train)
print(model.score(x_text,y_text))

print(x.head(3))
print(model.predict([[22,24.2,0,1,1,0,0,1]]))


