import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.externals import joblib
import json
import sys
data=pd.read_csv("NLPData/finalDataset.csv")
print(data)

x=data.as_matrix(columns= data.columns[1:8])
y=data.as_matrix(columns= data.columns[-1:])
y=np.squeeze(y);
print(x)
##splitting 70 - 30
from sklearn.cross_validation import train_test_split
xt,xtt,yt,ytt=train_test_split(x,y,test_size=0.3,random_state=3)

names = ['Topic Feature','ProperNoun Feature','Unknown Word Feature','Cue Word Feature','Bigram Feature','Tf-Idf','Sentence Position Feature']
'''
print('logistic regression')
lr=LogisticRegression(tol=1e-4, C=1.0, random_state=0, )
lr.fit(xt,yt)
ytestoutput=lr.predict(xtt)
print ('Accuracy',metrics.accuracy_score(ytt,ytestoutput))
print ('ROC',roc_auc_score(ytt,ytestoutput))
target_names = ['insummary', 'notinsummary']
print(classification_report(ytt,ytestoutput, target_names=target_names))
joblib.dump(lr, 'lr_model_save.pkl')

print('naive bayes')
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(xt,yt)
ypre=clf.predict(xtt)
print ('Accuracy',metrics.accuracy_score(ytt,ypre))
from sklearn.metrics import roc_auc_score
print ('ROC',roc_auc_score(ytt,ypre))
from sklearn.metrics import classification_report
target_names = ['insummary', 'notinsummary']
print(classification_report(ytt,ypre, target_names=target_names))
print ("")
joblib.dump(clf, 'GaussNB_model_save.pkl')

from sklearn.naive_bayes import BernoulliNB
clf = BernoulliNB()
clf.fit(xt,yt)
ypre=clf.predict(xtt)
print ('Accuracy',metrics.accuracy_score(ytt,ypre))
from sklearn.metrics import roc_auc_score
print ('Roc',roc_auc_score(ytt,ypre))
from sklearn.metrics import classification_report
target_names = ['insummary', 'notinsummary']
print(classification_report(ytt,ypre, target_names=target_names))
print ("")
joblib.dump(clf, 'BernoulliNB_model_save.pkl')

print('knn')
from sklearn.neighbors import KNeighborsClassifier
#neigh = KNeighborsClassifier(n_neighbors=5)
neigh = KNeighborsClassifier(n_neighbors=5)
neigh.fit(xt,yt)
ypre=neigh.predict(xtt)
print ('Accuracy',metrics.accuracy_score(ytt,ypre))
from sklearn.metrics import roc_auc_score
print ('Roc',roc_auc_score(ytt,ypre))
from sklearn.metrics import classification_report
taret_names = ['insummary', 'notinsummary']
print(classification_report(ytt,ypre, target_names=target_names))
joblib.dump(neigh, 'KNN_model_save.pkl')

print('svm')
from sklearn import svm
clf = svm.SVC()
clf.fit(xt, yt)
ypp=clf.predict(xtt)
print ('Accuracy',metrics.accuracy_score(ytt,ypp))
from sklearn.metrics import roc_auc_score
print ('Roc',roc_auc_score(ytt,ypp))
from sklearn.metrics import classification_report
target_names = ['insummary', 'notinsummary']
print(classification_report(ytt,ypp, target_names=target_names))
joblib.dump(clf, 'SVM_model_save.pkl')
'''
#random forest
print("Random Forest")
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier(n_estimators=900,random_state=0,max_depth=7)
rf.fit(xt, yt)
ytestoutput=rf.predict(xtt)
print ('Accuracy',metrics.accuracy_score(ytt,ytestoutput))
print ('ROC',roc_auc_score(ytt,ytestoutput))
target_names = ['insummary', 'notinsummary']
print(classification_report(ytt,ytestoutput, target_names=target_names))
print('features sorted by their score:')
print(sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), names),reverse=True))

joblib.dump(rf, 'RandomForest_model_save.pkl')

#adaboost
from sklearn.ensemble import AdaBoostClassifier
print('adaboost')
ab=AdaBoostClassifier(n_estimators=350,random_state=0,learning_rate= 0.9)
ab.fit(xt, yt)
ytestoutput=ab.predict(xtt)
print ('Accuracy',metrics.accuracy_score(ytt,ytestoutput))
print ('ROC',roc_auc_score(ytt,ytestoutput))
target_names = ['insummary', 'notinsummary']
print(classification_report(ytt,ytestoutput, target_names=target_names))
print('features sorted by their score:')
print(sorted(zip(map(lambda x: round(x, 4), ab.feature_importances_), names),reverse=True))

joblib.dump(ab, 'Adaboost_model_save.pkl')

#gradientboost
from sklearn.ensemble import GradientBoostingClassifier
print('gradboost')
gb=GradientBoostingClassifier(n_estimators=100,random_state=0,learning_rate= 0.1, loss='exponential')
gb.fit(xt, yt)
ytestoutput=gb.predict(xtt)
print ('Accuracy',metrics.accuracy_score(ytt,ytestoutput))
print ('ROC',roc_auc_score(ytt,ytestoutput))
target_names = ['insummary', 'notinsummary']
print(classification_report(ytt,ytestoutput, target_names=target_names))
print('features sorted by their score:')
featureWeightVector = sorted(zip(map(lambda x: round(x, 4), gb.feature_importances_), names),reverse=True)
print(featureWeightVector)
joblib.dump(gb, 'GradientBoost_model_save.pkl')


from sklearn.ensemble import GradientBoostingClassifier
print('gradboost')
gb=GradientBoostingClassifier(n_estimators=100,random_state=0,learning_rate= 0.05, loss='exponential', max_depth=5, )
gb.fit(xt, yt)
ytestoutput=gb.predict(xtt)
print ('Accuracy',metrics.accuracy_score(ytt,ytestoutput))
print ('ROC',roc_auc_score(ytt,ytestoutput))
target_names = ['insummary', 'notinsummary']
print(classification_report(ytt,ytestoutput, target_names=target_names))
print('features sorted by their score:')
print(sorted(zip(map(lambda x: round(x, 4), gb.feature_importances_), names),reverse=True))


from sklearn.ensemble import ExtraTreesClassifier
print('extratree')
etc=ExtraTreesClassifier(n_estimators=700,random_state=0,criterion='entropy', max_depth=7)
etc.fit(xt, yt)
ytestoutput=etc.predict(xtt)
print ('Accuracy',metrics.accuracy_score(ytt,ytestoutput))
print ('ROC',roc_auc_score(ytt,ytestoutput))
target_names = ['insummary', 'notinsummary']
print(classification_report(ytt,ytestoutput, target_names=target_names))
print('features sorted by their score:')
print(sorted(zip(map(lambda x: round(x, 4), etc.feature_importances_), names),reverse=True))
joblib.dump(etc, 'ExtraTreeClassifier_model_save.pkl')
print (ytestoutput)

file = open('NLPData/featureWeightVector.json', 'w+', encoding='utf-8')
json.dump(featureWeightVector, file)
file.close()