#%%
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import MinMaxScaler # Normalization
from sklearn.linear_model import LogisticRegression
import os
# %%
train_data=pd.read_csv("D:/Coding_Project/Python_KaggleChallenge_Titanic/Data/train.csv")
test_data=pd.read_csv("D:/Coding_Project/Python_KaggleChallenge_Titanic/Data/test.csv")
# %%
test_data.head()
# %%
train_data.head()
# %%
train_data.info()
# %%
test_data.info()
#%%
print('Number of siblings / spouses aboard the Titanic for Train Data: ',train_data.SibSp.unique())
print('Number of siblings / spouses aboard the Titanic for Test Data: ',test_data.SibSp.unique())
print('Ports of Embarkation for Train Data: ',train_data.Embarked.unique())
print('Ports of Embarkation for Test Data: ',test_data.Embarked.unique())
# %%
train_data = train_data.drop(['Name','Ticket','Cabin'], axis=1)
test_data = test_data.drop(['Name','Ticket','Cabin'], axis=1)
# %%
print('Duplicate elements in Train Data: ',train_data.duplicated().sum())
print('Duplicate elements in Test Data: ',test_data.duplicated().sum())
# %%
print('Null values in Train Data:\n',train_data.isnull().sum())
print('Null values in Test Data:\n',test_data.isnull().sum())
# %%
train_data.Sex=train_data.Sex.astype('category').cat.codes
test_data.Sex=test_data.Sex.astype('category').cat.codes
train_data.Embarked=train_data.Embarked.astype('category').cat.codes
test_data.Embarked=test_data.Embarked.astype('category').cat.codes
# %%
train_data['Age'].fillna(int(train_data['Age'].mean()), inplace=True)
test_data['Age'].fillna(int(test_data['Age'].mean()), inplace=True)
train_data['Embarked'].fillna(int(train_data['Embarked'].mean()), inplace=True)
test_data['Fare'].fillna(int(test_data['Fare'].mean()), inplace=True)
# %%
print('Null values in Train Data after preprocessing:\n',train_data.isnull().sum())
print('Null values in Test Data after preprocessing:\n',test_data.isnull().sum())
#%%
train_data.Age = MinMaxScaler().fit_transform(np.array(train_data.Age).reshape(-1,1))
train_data.Fare = MinMaxScaler().fit_transform(np.array(train_data.Fare).reshape(-1,1))
test_data.Age = MinMaxScaler().fit_transform(np.array(test_data.Age).reshape(-1,1))
test_data.Fare = MinMaxScaler().fit_transform(np.array(test_data.Fare).reshape(-1,1))
# %%
x = train_data.drop(['Survived'],axis=1)
# %%
y = train_data.Survived
# %%
clf = LogisticRegression(random_state=0,max_iter = 250).fit(x, y)
# %%
clf.score(x, y)
# %%
clf.predict(x)
# %%
output = clf.predict(test_data)
output
# %%
column1  = test_data.PassengerId
df = pd.DataFrame(column1, columns =['PassengerId'])
df1 = pd.DataFrame(output, columns =['Survived'])
result = pd.concat([df,df1],axis=1, join="inner")
result.info()
# %%
result.to_csv('Titanic_Survived_Predicion.csv',encoding='utf-8-sig')
# %%
