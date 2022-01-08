import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,recall_score,precision_score
from sklearn.model_selection import train_test_split,cross_val_score,RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.feature_selection import VarianceThreshold
from sklearn.svm import LinearSVC
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.datasets import make_classification

def with_scale(X_train, X_test, Y_train, Y_test):
	results = []
	clf = make_pipeline(StandardScaler(), svm.SVC())
	clf.fit(X_train, Y_train)
	results.append(100*accuracy_score(Y_test,clf.predict(X_test)).mean())

	model = make_pipeline(StandardScaler(), SGDClassifier(penalty = 'l1', alpha = 0.001, max_iter=10000, tol=1e-3, eta0=0.1))
	model.fit(X_train, Y_train)
	results.append(100*accuracy_score(Y_test,model.predict(X_test)).mean())

	rf_model = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators = 1000, random_state=1, class_weight='balanced'))
	rf_model.fit(X_train, Y_train)
	results.append(100*accuracy_score(Y_test,rf_model.predict(X_test)).mean())

	gb_model = make_pipeline(StandardScaler(),GradientBoostingClassifier(n_estimators=1000, random_state=1))
	gb_model.fit(X_train, Y_train)
	results.append(100*accuracy_score(Y_test,gb_model.predict(X_test)).mean())

	dt_model=make_pipeline(StandardScaler(),DecisionTreeClassifier())
	dt_model.fit(X_train,Y_train)
	results.append(100*accuracy_score(Y_test,dt_model.predict(X_test)).mean())

	ab_model = make_pipeline(StandardScaler(),AdaBoostClassifier())
	ab_model.fit(X_train,Y_train)
	results.append(100*accuracy_score(Y_test,ab_model.predict(X_test)).mean())
	return results

def without_scale(X_train, X_test, Y_train, Y_test):
	results = []

	clf = svm.SVC()
	clf.fit(X_train, Y_train)
	results.append(100*accuracy_score(Y_test,clf.predict(X_test)).mean())

	model = SGDClassifier(penalty = 'l1', alpha = 0.001, max_iter=10000, tol=1e-3, eta0=0.1)
	model.fit(X_train, Y_train)
	results.append(100*accuracy_score(Y_test,model.predict(X_test)).mean())

	rf_model = RandomForestClassifier(n_estimators = 1000, random_state=1, class_weight='balanced')
	rf_model.fit(X_train, Y_train)
	results.append(100*accuracy_score(Y_test,rf_model.predict(X_test)).mean())

	gb_model = GradientBoostingClassifier(n_estimators=1000, random_state=1)
	gb_model.fit(X_train, Y_train)
	results.append(100*accuracy_score(Y_test,gb_model.predict(X_test)).mean())

	dt_model = DecisionTreeClassifier()
	dt_model.fit(X_train,Y_train)
	results.append(100*accuracy_score(Y_test,dt_model.predict(X_test)).mean())

	ab_model = AdaBoostClassifier()
	ab_model.fit(X_train,Y_train)
	results.append(100*accuracy_score(Y_test,ab_model.predict(X_test)).mean())
	return results

def plots(res_with_1, res_without_1, res_with_2, res_without_2, res_with_3, res_without_3):

	methods = ["SVM", "SGD", "RF", "GB", "DT", "AB"]
	fig, axs = plt.subplots(3)
	fig.suptitle('Training Results')
	axs[0].set_xticks(range(len(methods))) # make sure there is only 1 tick per value
	axs[0].set_xticklabels(methods)
	x = axs[0].plot(methods, res_with_1, label = "with standard scaler", color="green")
	y = axs[0].plot(methods, res_without_1, label = "without standard scaler", color="red")
	axs[0].legend( ["with standard scaler", "without standard scaler"])
	axs[0].title.set_text("Without using Feature Selection and Outlier Detection")
	axs[1].set_xticks(range(len(methods))) # make sure there is only 1 tick per value
	axs[1].set_xticklabels(methods)
	x = axs[1].plot(methods, res_with_2, label = "with standard scaler", color="green")
	y = axs[1].plot(methods, res_without_2, label = "without standard scaler", color="red")
	axs[1].legend( ["with standard scaler", "without standard scaler"])
	axs[1].title.set_text("With using Feature Selection and Without using Outlier Detection")
	axs[2].set_xticks(range(len(methods))) # make sure there is only 1 tick per value
	axs[2].set_xticklabels(methods)
	x = axs[2].plot(methods, res_with_3, label = "with standard scaler", color="green")
	y = axs[2].plot(methods, res_without_3, label = "without standard scaler", color="red")
	axs[2].legend( ["with standard scaler", "without standard scaler"])
	axs[2].title.set_text("With using both Feature Selection and Outlier Detection")
	plt.legend(['with standard scalar', 'without standard scalar'], loc=0)
	plt.show()

data = pd.read_csv('student-por.csv')

data['romantic']=np.where(data['romantic'].values=='yes',1,0)
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

Y =  data['Dalc']
data.drop(['Dalc'],axis=1,inplace=True)
data.drop(['Mjob','Fjob','reason','guardian','Medu', 'Fedu', 'traveltime', 'famrel', 'failures','health'],axis=1,inplace=True)
X = data 

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2) 
print(1)
res_with_1 = with_scale(X_train,X_test,Y_train,Y_test)
res_without_1 = without_scale(X_train,X_test,Y_train,Y_test)

data.drop(['school', 'sex', 'address', 'famsize', 'Pstatus','schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic'],axis=1,inplace=True)

X = data 

X, Y = make_classification(n_samples=649, n_features=9, n_informative=9,
                           n_redundant=0, n_repeated=0, n_classes=5,
                           n_clusters_per_class=1, random_state=0)

estimator = RandomForestClassifier() #SVC(kernel="linear")

min_features_to_select = 1  # Minimum number of features to consider
rfecv = RFECV(estimator=estimator, step=1, cv=StratifiedKFold(2),
              scoring='accuracy',
              min_features_to_select=min_features_to_select)
rfecv.fit(X, Y)

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2) 
print(2)
res_with_2 = with_scale(X_train,X_test,Y_train,Y_test)
res_without_2 = without_scale(X_train,X_test,Y_train,Y_test)

ee = EllipticEnvelope(contamination=0.1)    #75
yhat = ee.fit_predict(X)
mask = yhat != -1
X, Y = X[mask, :], Y[mask]

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2) 
print(3)
res_with_3 = with_scale(X_train,X_test,Y_train,Y_test)
res_without_3 = without_scale(X_train,X_test,Y_train,Y_test)
print(4)

plots(res_with_1, res_without_1, res_with_2, res_without_2, res_with_3, res_without_3)