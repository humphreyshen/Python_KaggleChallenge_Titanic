#%%
import seaborn as sns
import numpy as np 
import pandas as pd 
from sklearn.preprocessing import MinMaxScaler # Normalization
from sklearn import preprocessing
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
# %%
train_data=pd.read_csv("D:/Coding_Project/Python_KaggleChallenge_Titanic/Data/train.csv")
test_data=pd.read_csv("D:/Coding_Project/Python_KaggleChallenge_Titanic/Data/test.csv")
# %%
train_data.head()
# %%
sns.heatmap(train_data.corr(),cmap='YlGnBu')
# %%
train_data = train_data.drop(['Name','Ticket','Cabin'], axis=1)
test_data = test_data.drop(['Name','Ticket','Cabin'], axis=1)

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
#%%
train_data
# %%
x = train_data.copy()
x=np.array(x.drop(['Survived'],1).astype(float))
x=preprocessing.scale(x)
y = np.array(train_data['Survived'])
clf=KMeans(n_clusters=2)
clf.fit(x)
correct = 0.
for i in range(len(x)):
    predict_me = np.array(x[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = clf.predict(predict_me)
    if prediction[0] == y[i]:
        correct += 1

print(correct/len(x))
#%%
print(clf.labels_)
#%%
plt.scatter(train_data['PassengerId'],y=train_data['Sex'],c=clf.labels_.astype(float))
# %%
plt.scatter(train_data['PassengerId'],y=train_data['Fare'],c=clf.labels_.astype(float))
# %%
