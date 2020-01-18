
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv("50_Startups_EDA.csv")

dataset.isnull()

dataset.isnull().sum()

x= dataset.iloc[:,0:3].values
y= dataset.iloc[:,3:4].values

from sklearn.preprocessing.imputation import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])

# %%
