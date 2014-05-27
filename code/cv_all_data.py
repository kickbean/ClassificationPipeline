'''
Created on Apr 28, 2014

@author: Songfan
'''

from __future__ import print_function

import csv
import numpy as np
import pylab as pl
import sklearn.cross_validation as cv
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.svm import SVC


print(__doc__)
method = 3
if method == 1: file_name = 'feat_first_frame'
elif method == 2: file_name = 'feat_dynamic_warp_mean'
elif method == 3: file_name = 'feat_dense_flow_mean'

X = csv.reader(open('../data/'+file_name+'.csv'), delimiter=',')
X = np.array(list(X)).astype(np.float)
data_num, dim = X.shape
#print X.shape, type(X[3][3])

y = csv.reader(open('../data/label.csv'), delimiter='.')
y = np.squeeze(np.array(list(y)).astype(np.int))
#pl.hist(y)
#pl.show()

clf = SVC(kernel='linear')
kf = cv.KFold(data_num,n_folds=20)
y_pred = np.ones(data_num)
for train_index,test_index in kf:
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    y_pred[test_index] = clf.fit(X_train,y_train).predict(X_test)
    
print(metrics.classification_report(y, y_pred))
print("Confusion matrix")
print(metrics.confusion_matrix(y, y_pred))
print()
# write prediction to file
#np.savetxt('../result/'+file_name+'_pred.csv',y_pred,delimiter=',')

#print()
#print("mis-classification examples")
#label_set = np.unique(y)
#for i in label_set:
#    for j in label_set:
#        if i != j:
#            print("true class: %d, predicted class: %d" % (i,j))
#            mis_cls = np.intersect1d(y[np.where(y==i)],y_pred[np.where(y_pred==j)])
#            print(mis_cls)
        
        
        
        
        