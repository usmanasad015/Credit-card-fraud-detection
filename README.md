# Credit-card-fraud-detection
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn import metrics

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier
data = pd.read_csv("D:\Semester7\dataMining\semsterProject\creditcard.csv")
data

X = data.drop(columns='Class')
Y = data[['Class']]
x
X
Y
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.4)
# Random Forest Classifier


clf = RandomForestClassifier()
clf.fit(X_train,Y_train)
Y_prediction = clf.predict(X_test)
print("Accuracy", metrics.accuracy_score(Y_test, Y_prediction))
## Decison tree classifier
dt = DecisionTreeClassifier()
dt.fit(X_train, Y_train)
pred = dt.predict(X_test)
print("Accuracy:", metrics.accuracy_score(Y_test, pred))

# Logistic Regression
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(X_train,Y_train)
lr_prd = logreg.predict(X_test)
print("Accuracy:", metrics.accuracy_score(Y_test,lr_prd))
#KNN
from sklearn.neighbors import KNeighborsClassifier
knc = KNeighborsClassifier(n_neighbors=17)
knc.fit(X_train,Y_train)
knc_pred = knc.predict(X_test)
print("Accuracy:", metrics.accuracy_score(Y_test,knc_pred))
#Naive Bayes
from sklearn.naive_bayes import GaussianNB
gNB = GaussianNB()
gNB.fit(X_train,Y_train)
gNB_pred = gNB.predict(X_test)

print("Accuracy:", metrics.accuracy_score(Y_test,gNB_pred))
from sklearn.svm import SVC

svc = SVC(kernel='linear')
svc.fit(X_train,Y_train)
svc_prd = svc.predict(X_test)
print("Accuracy:", metrics.accuracy_score(Y_test,svc_prd))
