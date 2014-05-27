'''
Created on Apr 24, 2014

@author: Songfan
'''
from __future__ import print_function

import csv
import numpy as np
import pylab as pl
import sklearn.cross_validation as cv
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


print(__doc__)

# Loading the Digits dataset
#digits = datasets.load_digits()

# To apply an classifier on this data, we need to flatten the image, to
# turn the data in a (samples, feature) matrix:
#n_samples = len(digits.images)
#X = digits.images.reshape((n_samples, -1))
#y = digits.target

X = csv.reader(open('../data/feature_harris.csv'), delimiter=',')
X = np.array(list(X)).astype(np.float)
data_num, dim = X.shape
#print X.shape, type(X[3][3])

y = csv.reader(open('../data/label.csv'), delimiter='.')
y = np.squeeze(np.array(list(y)).astype(np.int))
#pl.hist(y)
#pl.show()

# Split the dataset in two equal parts
X_train, X_test, y_train, y_test = cv.train_test_split(
    X, y, test_size=0.3, random_state=0)

# Set the parameters by cross-validation
tuned_paras = {'max_depth': (1,2)}

scores = ['f1'] # ['precision', 'recall']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(RandomForestClassifier(max_features=400,n_estimators=50), tuned_paras, cv=5, scoring=score)
#    clf = RandomForestClassifier(n_estimators=2,max_depth=2)
    clf.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_estimator_)
    print()
    print("Grid scores on development set:")
    print()
    for params, mean_score, scores in clf.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r"
              % (mean_score, scores.std() / 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(X_test)
    print(metrics.classification_report(y_true, y_pred))
    print()
    print("Confusion matrix")
    print(metrics.confusion_matrix(y_test, y_pred))
    
# final cross validation on the entire dataset
clf_best = clf
kf = cv.KFold(data_num,n_folds=5)
y_pred = np.ones(data_num)
for train_index,test_index in kf:
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    y_pred[test_index] = clf_best.fit(X_train,y_train).predict(X_test)
    
print(metrics.classification_report(y, y_pred))

#def sf_cross_val_clf(X,y,tuned_para):