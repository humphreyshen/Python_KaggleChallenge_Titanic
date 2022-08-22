#%%
import seaborn as sns
import numpy as np 
import pandas as pd 
from sklearn.preprocessing import MinMaxScaler # Normalization
from sklearn import preprocessing
from sklearn.cluster import KMeans
import plotly.express as px
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
fig = px.scatter(train_data,x='PassengerId',y='Fare',color=clf.labels_)
fig.show()

# %%
test_x=test_data.copy()
scaled_test=preprocessing.scale(test_x)
output = clf.predict(scaled_test)
output
#%%
column1  = test_data.PassengerId
df = pd.DataFrame(column1, columns =['PassengerId'])
df1 = pd.DataFrame(output, columns =['Survived'])
result = pd.concat([df,df1],axis=1, join="inner")
visualized=test_data
merge=visualized.merge(result, how='inner', on='PassengerId')
merge.Survived=merge.Survived.astype('object')
fig = px.scatter(merge, x="Age", y="Fare", color="Survived")
fig.show()
fig.write_image("Prediction_KMeans.png")

# %%
result.to_csv('Prediction_KMeans.csv',encoding='utf-8-sig',index=False)