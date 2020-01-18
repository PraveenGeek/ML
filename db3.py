#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as ptd

dataset = pd.read_csv("50_startups.csv")

dataset.isnull().sum()

x= dataset.iloc[:,:-1].values
y= dataset.iloc[:,4].values

from sklearn.preprocessing import LabelEncoder , OneHotEncoder
le_x = LabelEncoder ()
x[:,3] = le_x.fit_transform(x[:,3])
one_x = OneHotEncoder(categorical_features =[3])
x = one_x.fit_transform(x).toarray () 
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_text = train_test_split(x, y, test_size=.30, random_state=0) 
from sklearn.linear_model import LinearRegression
regress = LinearRegression()
regress.fit(x_train,y_train)
Y_pred = regress.predict(x_test)
regress.score(x_train,y_train) 

# %%
