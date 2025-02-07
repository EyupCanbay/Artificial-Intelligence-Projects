import pandas as pd
import numpy as nm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv("./datasets/banka3.csv")
df = df[["sorgu", "label"]]
#print(df.head(3))
stopwords=['fakat','lakin','ancak','acaba', 'ama', 'aslında', 'az', 'bazı', 'belki', 'biri', 'birkaç', 'birşey', 'biz', 'bu', 'çok', 'çünkü', 'da', 'daha', 'de', 'defa', 'diye', 'eğer', 'en', 'gibi', 'hem', 'hep', 'hepsi', 'her', 'hiç', 'için', 'ile', 'ise', 'kez', 'ki', 'kim', 'mı', 'mu', 'mü', 'nasıl', 'ne', 'neden', 'nerde', 'nerede', 'nereye', 'niçin', 'niye', 'o', 'sanki', 'şey', 'siz', 'şu', 'tüm', 've', 'veya', 'ya', 'yani']

mesaj = input("Yapmak isteidğiniz işlemi giriniz: ")
mesajdf = pd.DataFrame({"sorgu": mesaj, "label": 0},index = [34])
df = pd.concat([df, mesajdf], ignore_index=True)

for word in stopwords:
    word = " " + word + " "
    df["sorgu"] = df["sorgu"].str.replace(word, " ")

#print(df.head(3))
cv = CountVectorizer(max_features=50)
x = cv.fit_transform(df["sorgu"]).toarray()
y = df["label"]
# print(x[0])S
tahmin = x[-1].copy()

x=x[0:-1]
y=y[0:-1]
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=21,train_size=0.7)


rf = RandomForestClassifier()
model = rf.fit(x_train,y_train)
skor = model.score(x_test,y_test)

sonuc = model.predict([tahmin])

print("Sonuç: ", skor, "Tahmin: ", sonuc)
