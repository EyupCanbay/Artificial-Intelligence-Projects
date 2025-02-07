import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


df = pd.read_csv('./datasets/audi_a1.csv')


df=df.drop(columns=['index','href','MileageRank', 'PriceRank', 'PPYRank','Score'])

print(df.head(3))
print(df.info())


df.columns = ["yil", "kasa", "mil", "motor", "ps", "vites", "yakit", "sahip", "fiyat", "ppy"]

df['motor'] = df['motor'].str.replace('L','')
df['motor'] = pd.to_numeric(df['motor'])

df =pd.get_dummies(df,columns=['kasa', 'vites', 'yakit'], dtype=int, drop_first=True)
print(df.head(3))

lm = LinearRegression()
y = df['fiyat']
x = df.drop("fiyat", axis=1) 
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.70, random_state=22)

lm = LinearRegression()
model=lm.fit(x_train,y_train)
predictModel = model.predict([[2016, 30000, 1.0,90, 5, 2600, 0, 1]])
print(predictModel)

print(model.score(x_test,y_test))