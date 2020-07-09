import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import seaborn as sns
from math import sqrt

df = pd.read_csv('kc_house_data.csv')
print(df.head())

features = ['sqft_living', 'bathrooms', 'bedrooms', 'sqft_lot', 'floors', 'zipcode']
x = df[features]
y = df['price']

print(x)
sns.pairplot(df,x_vars=features ,y_vars="price", kind = 'reg')
plt.show()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=2)

model = LinearRegression()
model.fit(x_train, y_train)
pred = model.predict(x_test)
acc = model.score(x_test, y_test)
print("accuracy = ", acc)

rms = sqrt(mean_squared_error(y_test, pred))
print(rms)

plt.plot(x_test, y_test, '.', x_test, pred, '-')
plt.show()

sns.boxplot(x = df['zipcode'], y = df['price'], data=df)
plt.show()

house1 = df[df['id']== 5309101200]
print(house1['price'])
print(model.predict(house1[features]))

house2 = df[df['id']== 1925069082]
print(house2['price'])
print(model.predict(house2[features]))

