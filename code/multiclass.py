'''
Multiclass classification


Created on Apr 23, 2014

@author: Songfan
'''
import csv
import numpy as np
import pylab as pl
from sklearn.svm import SVC
from sklearn import cross_validation
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import classification_report
from sklearn.grid_search import GridSearchCV

# read data
X = csv.reader(open('../data/feature_intensity.csv'), delimiter=',')
X = np.array(list(X)).astype(np.float)
data_num, dim = X.shape
print X.shape, type(X[3][3])

y = csv.reader(open('../data/label.csv'), delimiter='.')
y = np.squeeze(np.array(list(y)).astype(np.int))
print y.shape, type(y[3])

# split data to train and test
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=0)

# Kfold classification
tune_para = [{'kernel': ['rbf'], 'gamma':[1e-3, 1e-4], 'C':[1,10,100,1000]},
             {'kernel': ['linear'], 'C':[1,10,100,1000]}]

clf = GridSearchCV(SVC(C=1), tune_para, cv=5, scoring='f1')
clf.fit(X_train, y_train)
print('Best parameters set found on development set')
print(clf.best_estimator_)
print()
print('Grid scores on development set')
for params, mean_score, scores in clf.grid_scores_:
    print("0.3f (+/-%0.3f) for %r"
          % (mean_score, scores.std()/2, params))

print()
print("Detailed classification report:")
print()
print("The model is trained on the full development set.")
print("The scores are computed on the full evaluation set.")
print()
y_true, y_pred = y_test, clf.predict(X_test)
print(classification_report(y_true, y_pred))
print()