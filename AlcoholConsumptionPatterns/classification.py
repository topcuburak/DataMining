import pandas as pd
import os

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
import numpy as np 

from sklearn.metrics import accuracy_score,confusion_matrix,recall_score,precision_score


from sklearn.model_selection import train_test_split,cross_val_score,RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

#plt.rcParams['figure.figsize'] = (10,6)

data =pd.read_csv('student-por.csv')

data['Dalc'] = data['Dalc']+data['Walc']

data['school'] = np.where(data['school'].values=='GP',1,0)
data['sex'] = np.where(data['sex'].values=='M',1,0)
data['address'] = np.where(data['address'].values=='U',1,0)
data['famsize'] = np.where(data['famsize'].values=='GT3',1,0)
data['Pstatus'] = np.where(data['Pstatus'].values=='A',1,0)
data.drop(['Mjob', 'Fjob', 'guardian', 'nursery', 'reason', 'Walc'], axis=1)
data['schoolsup'] = np.where(data['schoolsup'].values=='yes',1,0)
data['famsup'] = np.where(data['famsup'].values=='yes',1,0)
data['paid'] = np.where(data['paid'].values=='yes',1,0)
data['higher'] = np.where(data['higher'].values=='yes',1,0)
data['internet'] = np.where(data['internet'].values=='yes',1,0)
data['romantic'] = np.where(data['romantic'].values=='yes',1,0)


print(data["school"])
print(data["sex"])
print(data["age"])
print(data["address"])
print(data["famsize"])
print(data["Pstatus"])
print(data["Medu"])
print(data["Fedu"])
print(data["traveltime"])
print(data["studytime"])
print(data["failures"])
print(data["schoolsup"])
print(data["famsup"])
print(data["paid"])
print(data["activities"])
print(data["activities"])
print(data["activities"])
print(data["activities"])
print(data["activities"])
print(data["activities"])
print(data["activities"])
print(data["activities"])
print(data["activities"])


#data = data.values


X = data.drop('Dalc',axis=1)
y = data['Dalc']

print(X.shape)
print(y.shape)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=20)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

models=[]
names=['Decision Tree','SVC','Random Forest','Adaboost','XGB classifier']
cv_models=[]
# Decision Tree
dt=[]
dt=DecisionTreeClassifier()
dt.fit(X_train,y_train)
dt_pred=dt.predict(X_test)
models.append(accuracy_score(y_test,dt_pred))

cv_models.append(cross_val_score(dt,X,y,cv=5).mean())

#svm
svc=SVC()
svc.fit(X_train,y_train)
svc_pred = svc.predict(X_test)
models.append(accuracy_score(y_test,svc_pred).mean())
cv_models.append(cross_val_score(svc,X,y,cv=5).mean())

#Random Forest
rf =RandomForestClassifier()
rf.fit(X_train,y_train)
rf_pred = rf.predict(X_test)
models.append(accuracy_score(y_test,rf_pred))
cv_models.append(cross_val_score(rf,X,y,cv=5).mean())

#AdaBoost
ab = AdaBoostClassifier()
ab.fit(X_train,y_train)
ab_pred = ab.predict(X_test)
models.append(accuracy_score(y_test,ab_pred))
cv_models.append(cross_val_score(ab,X,y,cv=5).mean())
print(cross_val_score(ab,X,y,cv=5).mean())

xgb =XGBClassifier()
xgb.fit(X_train,y_train)
xgb_pred =xgb.predict(X_test)
models.append(accuracy_score(y_test,xgb_pred))
cv_models.append(cross_val_score(xgb,X,y,cv=5).mean())
print(cross_val_score(xgb,X,y,cv=5).mean())


cv_models=[i*1000 for i in cv_models]
models = [i*1000 for i in models]

# final models dataset
final_df=pd.DataFrame({'Model_names':names,'Train_test_split_score in %':models,'CV_Score in %':cv_models})
print(final_df)


'''data.isnull().sum()

data.describe().T.style.bar(subset=['mean'])\
                            .background_gradient(subset=['std'])\
                            .background_gradient(subset=['50%'])\
                            .background_gradient(subset=['max'])

'''

'''# lets seperate the categorical variable and numerical variables
cat_col = [x for x in data.columns if data[x].dtypes=='O']
num_col = [x for x in data.columns if x not in cat_col]

# Target Variable
plt.figure(figsize=(12,6))
sns.countplot(x='G_Total', data=data,
                   facecolor=(0,0,0,0),
                   linewidth=5,
                   edgecolor=sns.color_palette("dark", 10))

plt.figure(figsize=(10,6))
sns.countplot(data=data,x='age',hue='Dalc')
plt.tight_layout()
plt.show()

# lets examine various features together
plt.figure(figsize=(30,30))
feature = [x for x in num_col if 'G_Total' not in x]
for i in enumerate(feature):
    plt.subplot(5,4,i[0]+1)
    sns.countplot(i[1],hue='Dalc',data=data)
    plt.title(i[1]+' vs workday alcohol Consumption (Dalc)')
    plt.xticks(rotation=45)
'''
# school vs alcohol consumption on workday
'''sns.countplot(data=data,x='school',hue='Dalc')

# female vs alcohol consumption on weekends
sns.countplot(data=data,x='sex',hue='Dalc',palette='Set3')

# Countplot provide us the count values
sns.countplot('Dalc',data=data,palette='winter')

#grade vs Alcohol consumption
sns.catplot(y="G_Total", x="Dalc",hue='sex', kind="swarm", data=data)

# alcohol consumption vs grades(under average or above average)
average=data['G_Total'].mean()
data['average'] = ['under average'if i < average else 'above average' for i in data.G_Total]
sns.swarmplot(x='Dalc',y='G_Total',hue='average',data=data,palette={'above average':'Red','under average':'green'})

# Dalc vs G_Total on gender
sns.catplot(x='Dalc',y='G_Total',hue='school',col='sex',data=data,kind='bar')

#countplot for romantic vs Dalc
sns.catplot(x='romantic',hue='Dalc',data=data,kind='count')

sns.catplot(x='Dalc',y='G_Total',col='romantic',data=data,kind='bar',palette='summer')

'''

'''plt.figure(figsize=(30,30))
for i in enumerate(cat_col):
    plt.subplot(5,4,i[0]+1)
    sns.countplot(x=i[1],hue='Dalc',data=data,palette='nipy_spectral')
'''

GP = data[data.school == 'GP']
MS = data[data.school == 'MS']

'''fig,ax=plt.subplots(1,2,figsize=(12,6))
sns.kdeplot(GP.G_Total,label="GP",ax=ax[0])
sns.kdeplot(MS.G_Total,label="MS",ax=ax[0])
'''
'''sns.kdeplot(GP.Dalc,label='GP',ax=ax[1],)
sns.kdeplot(MS.Dalc,label='MS',ax=ax[1])
ax[0].set_ylabel('')
ax[1].set_ylabel('')

plt.show()

plt.figure(figsize=(16,6))
sns.heatmap(data.corr(),annot=True,cmap='cividis')
'''

'''data['romantic']=np.where(data['romantic'].values=='yes',1,0)
data['internet']=np.where(data['internet'].values=='yes',1,0)
data['higher']=np.where(data['higher'].values=='yes',1,0)
data['nursery']=np.where(data['nursery'].values=='yes',1,0)
data['activities']=np.where(data['activities'].values=='yes',1,0)
data['paid']=np.where(data['paid'].values=='yes',1,0)
data['famsup']=np.where(data['famsup'].values=='yes',1,0)
data['schoolsup']=np.where(data['schoolsup'].values=='yes',1,0)
data['school']=np.where(data['school'].values=='GP',1,0)
data['sex']=np.where(data['sex'].values=='M',1,0)
data['address']=np.where(data['address'].values=='U',1,0)
data['famsize']=np.where(data['famsize'].values=='GT3',1,0)
data['Pstatus']=np.where(data['Pstatus'].values=='A',1,0)

Mjob=pd.get_dummies(data['Mjob'],prefix='M_job',drop_first=True)
Fjob=pd.get_dummies(data['Fjob'],prefix='F_job',drop_first=True)
reason = pd.get_dummies(data['reason'],prefix='reason',drop_first=True)
guardian = pd.get_dummies(data['guardian'],prefix='guardian',drop_first=True)

data.drop(['Mjob','Fjob','reason','guardian','G1','G2','G3','average'],axis=1,inplace=True)

df_1 = pd.concat([data,Mjob,Fjob,reason,guardian],axis=1)
df_1.head()

X=df_1.drop('G_Total',axis=1)
y=df_1['G_Total']

print(X.shape)
print(y.shape)

'''
