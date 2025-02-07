#KREDİ KARTI EXTRESİNİ ÖDEYECEK Mİ ÖDEMEYECEK Mİ TAHMİNİ
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv("./datasets/credit_Card.csv")
print(df.head(3))

df = df.drop(["ID"], axis=1)
y = df["default.payment.next.month"]
x = df.drop(["default.payment.next.month"],axis=1)

x_train, x_text, y_train, y_text = train_test_split(x,y,test_size=0.8, random_state=50)

log = LogisticRegression()
model = log.fit(x_train, y_train)

print(model.score(x_text,y_text))

deneme_x=np.array(x.iloc[1903])
print(deneme_x)

print(model.predict([deneme_x]))
print(y.iloc[1903])
