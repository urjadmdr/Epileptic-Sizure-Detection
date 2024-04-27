import pandas as pd
import numpy as np
import collections

header_names =['X1','X2','X3','X4','X5','X6','X7','X8','X9','X10','X11','X12','X13','X14','X15','X16','X17','X18','X19','X20','X21','X22','X23','X24','X25','Y']
data= pd.read_csv('EEG.csv',header=None, names=header_names)

feature_set= data[['X1','X2','X3','X4','X5','X6','X7','X8','X9','X10','X11','X12','X13','X14','X15','X16','X17','X18','X19','X20','X21','X22','X23','X24','X25']]

#independent variable
X= np.asarray(feature_set)

#dependent variable
y= np.asarray(data['Y'])

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=4)

logreg = LogisticRegression()
#rfe = RFE(logreg, 10)
rfe = logreg.fit(X_train, y_train)
y_predict = rfe.predict(X_test)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_predict))

print(collections.Counter(y_predict))
print(collections.Counter(y_test))