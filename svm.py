import pandas as pd
import numpy as np
import collections
from sklearn import preprocessing

header_names =['mean','pstdev','pvariance','stdev','variance','kurtosis','skew','min','max','label']
data= pd.read_csv('FEATURE_SET.csv',header=None, names=header_names)
print(data)

feature_set= data[['mean','pstdev','pvariance','stdev','variance','kurtosis','skew','min','max']]

#independent variable
X= np.asarray(feature_set)

#dependent variable
y= np.asarray(data['label'])

#X = preprocessing.scale(X)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=4)

from sklearn import svm
classifier = svm.SVC(kernel='linear', gamma='auto', C=2)
svm_model= classifier.fit(X_train, y_train)

y_predict = classifier.predict(X_test)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_predict))

print(collections.Counter(y_predict))
print(collections.Counter(y_test))