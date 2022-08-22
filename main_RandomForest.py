#%%
import pandas as pd
import numpy as np 
import plotly.express as px
from sklearn import tree 
from IPython.display import Image
import pydot
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
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
features=list(train_data.columns[2:])
features
# %%
y=train_data['Survived']
x=train_data[features]
clf=RandomForestClassifier(n_estimators=10)
clf=clf.fit(x,y)

#%%
correct = 0.
x_1=np.array(train_data.drop(['PassengerId'],1).drop(['Survived'],1).astype(float))
y = np.array(train_data['Survived'])
for i in range(len(x_1)):
    predict_me = np.array(x_1[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = clf.predict(predict_me)
    if prediction[0] == y[i]:
        correct += 1

print(correct/len(x))
# %%
test_features=list(test_data.columns[1:])
test_data_x=test_data[test_features]
prediction=clf.predict(test_data_x)
# %%
column1  = test_data.PassengerId
df = pd.DataFrame(column1, columns =['PassengerId'])
df1 = pd.DataFrame(prediction, columns =['Survived'])
result = pd.concat([df,df1],axis=1, join="inner")
visualized=test_data
merge=visualized.merge(result, how='inner', on='PassengerId')
merge.Survived=merge.Survived.astype('object')
fig = px.scatter(merge, x="PassengerId", y="Fare", color="Survived")
fig.show()
fig.write_image("Prediction_RandomForest.png")

# %%
result.to_csv('Prediction_RandomForest.csv',encoding='utf-8-sig',index=False)
# %%
