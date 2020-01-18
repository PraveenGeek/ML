#%%
import pandas as pd 
import matplotlib.pyplot as plt
dataset=pd.read_csv("height_weight.csv")
x=dataset.iloc[:,0:-1].values 
y=dataset.iloc[:,1:2].values
dataset.fillna(dataset.mean(),inplace=True)
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_text = train_test_split(x, y, test_size=.20, random_state=0)
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(x_train,y_train)
Y_pred = reg.predict(x_test)
my_H=[[180]]
print(reg.predict(my_H))
plt.title("simple linear regression")
plt.xlabel("Height")
plt.ylabel("Weight")
plt.scatter(x_train,y_train,color='red')
plt.plot(x_train,reg.predict(x_train),color='blue')
plt.show()
# %%
