#ÖĞRENCİNİN ÇALIŞMASINI SCORE DEĞERİNİ TAHMİN ETME
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge

df = pd.read_csv("./datasets/student_scores.csv")
#print(df.head(3))    

y = df["Scores"]
x = df[["Hours"]]

plt.style.use("fivethirtyeight")
plt.figure(figsize=(6,6))
plt.scatter(x,y)
plt.show()
lm = LinearRegression()
model = lm.fit(x,y)
print("Model Score: ",model.score(x,y))

alfalar=[1,20,30, 100, 200]
for i in alfalar:
    r = Ridge(alpha = i)
    modelR = r.fit(x,y)
    print(f"while alpha is {i}, ridge score: {modelR.score(x,y)} , model's coef: {modelR.coef_} , model's intercept: {modelR.intercept_}")
