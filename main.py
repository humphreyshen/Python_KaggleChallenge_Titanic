#%%
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import MinMaxScaler # Normalization
from sklearn.linear_model import LogisticRegression
import os
# %%
train_data=pd.read_csv("D:\Coding_Project\Python_KaggleChallenge_Titanic\Data\train.csv")
test_data=pd.read_csv("D:\Coding_Project\Python_KaggleChallenge_Titanic\Data\test.csv")
# %%
