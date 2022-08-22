#%%
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import MinMaxScaler # Normalization
from sklearn.linear_model import LogisticRegression
import plotly.express as px
# %%
train_data=pd.read_csv("D:/Coding_Project/Python_KaggleChallenge_Titanic/Data/train.csv")
test_data=pd.read_csv("D:/Coding_Project/Python_KaggleChallenge_Titanic/Data/test.csv")

train_data.info()
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
origin_train_data=train_data.copy()
origin_test_data=test_data.copy()
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
result
# %%
visualized=origin_test_data
visualized
# %%
merge=visualized.merge(result, how='inner', on='PassengerId')
# %%
merge.Survived=merge.Survived.astype('object')
# %%
merge.dtypes
#%%
fig = px.scatter(merge, x="Age", y="Fare", color="Survived")
fig.show()
fig.write_image("Prediction_LogisticRegression.png")
# %%
result.to_csv('Prediction_LogisticRegression.csv',encoding='utf-8-sig',index=False)
# %%
