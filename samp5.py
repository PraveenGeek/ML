import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
dataset=pd.read_csv("50_startups.csv")
dataset.isnull().sum()

x=dataset.iloc[:,0:-1].values
y=dataset.iloc[:,4:5].values

le_x=LabelEncoder()
x[:,3] = le_x.fit_transform(x[:,3])
one_x=OneHotEncoder(categorical_features=[3])
x=one_x.fit_transform(x).toarray()

x=x[:,1:]

#dataset.fillna(dataset.mean(), inplace=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
regress=LinearRegression()
regress.fit(x_train,y_train)
Y_pred=regress.predict(x_test)

regress.score(x_train,y_train)
regress.score(x_test,y_test)

import statsmodels.api as sm 
x = np.append(arr = x, values = np.ones(shape = (50,1), dtype = int) , axis = 1) 

x_ov = x[:,[0,1,2,3,4,5]] 
regress_ols = sm.OLS(endog = y , exog = x_ov).fit() 
regress_ols.summary()


x_ov = x[:,[0,1,3,4,5]] 
regress_ols = sm.OLS(endog = y , exog = x_ov).fit() 
regress_ols.summary()


x_ov = x[:,[0,3,4,5]] 
regress_ols = sm.OLS(endog = y , exog = x_ov).fit() 
regress_ols.summary()


x_ov = x[:,[0,3,5]] 
regress_ols = sm.OLS(endog = y , exog = x_ov).fit() 
regress_ols.summary()


x_ov = x[:,[0,3]] 
regress_ols = sm.OLS(endog = y , exog = x_ov).fit() 
regress_ols.summary()