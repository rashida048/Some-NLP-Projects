# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 16:25:50 2021

@author: rashi
"""

import pandas as pd
df = pd.read_csv('amazon_baby.csv')
df.head()

print(len(df) - len(df.dropna()))

df = df.dropna()

df['rating'].value_counts(normalize=True)

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
df['rating'].value_counts().plot(kind='bar')
plt.xlabel("Rting")
plt.ylabel("Count")
plt.title("Number of Data Count for Each Rating")


import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))


import numpy as np
np.random.seed(34)
df1 = df.sample(frac = 0.3)

df1['rating'].value_counts(normalize=True)

print("length of the dataset used in the project")
print(len(df1))

plt.figure(figsize=(8, 6))
df1['rating'].value_counts().plot(kind='bar')
plt.xlabel("Rting")
plt.ylabel("Count")
plt.title("Number of Data Count for Each Rating")

for i in df1['review']:
    i = str(i)
    
df1['sentiments'] = df1.rating.apply(lambda x: 0 if x in [1, 2] else 1)
#print(df1['sentiments'])

df1['sentiments'].value_counts(normalize=True)

X = df1['review']
y = df1['sentiments']


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                        test_size = 0.5, random_state=24)


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
ctmTr = cv.fit_transform(X_train)
X_test_dtm = cv.transform(X_test)

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(ctmTr, y_train)

lr_score = lr.score(X_test_dtm, y_test)
print("Results for Logistic Regression with CountVectorizer")
print(lr_score)

y_pred_lr = lr.predict(X_test_dtm)
from sklearn.metrics import confusion_matrix

#Confusion matrix
cm_lr = confusion_matrix(y_test, y_pred_lr)

tn, fp, fn, tp = confusion_matrix(y_test, y_pred_lr).ravel()
print(tn, fp, fn, tp)

tpr_lr = round(tp/(tp + fn), 4)
tnr_lr = round(tn/(tn+fp), 4)

print(tpr_lr, tnr_lr)





from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                        test_size = 0.5, random_state=123)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
ctmTr = cv.fit_transform(X_train)
X_test_dtm = cv.transform(X_test)

from sklearn import svm

svcl = svm.SVC()
svcl.fit(ctmTr, y_train)
svcl_score = svcl.score(X_test_dtm, y_test)
print("Results for Support Vector Machine with CountVectorizer")
print(svcl_score)

y_pred_sv = svcl.predict(X_test_dtm)

#Confusion matrix
cm_sv = confusion_matrix(y_test, y_pred_sv)

tn, fp, fn, tp = confusion_matrix(y_test, y_pred_sv).ravel()
print(tn, fp, fn, tp)

tpr_sv = round(tp/(tp + fn), 4)
tnr_sv = round(tn/(tn+fp), 4)

print(tpr_sv, tnr_sv)





from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                        test_size = 0.5, random_state=143)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
ctmTr = cv.fit_transform(X_train)
X_test_dtm = cv.transform(X_test)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(ctmTr, y_train)

knn_score = knn.score(X_test_dtm, y_test)
print("Results for KNN Classifier with CountVectorizer")
print(knn_score)

y_pred_knn = knn.predict(X_test_dtm)

#Confusion matrix
cm_knn = confusion_matrix(y_test, y_pred_knn)

tn, fp, fn, tp = confusion_matrix(y_test, y_pred_knn).ravel()
print(tn, fp, fn, tp)

tpr_knn = round(tp/(tp + fn), 4)
tnr_knn = round(tn/(tn+fp), 4)

print(tpr_knn, tnr_knn)




from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                        test_size = 0.5, random_state=45)

from sklearn.feature_extraction.text import TfidfVectorizer

#tfidf vectorizer

vectorizer = TfidfVectorizer()

X_train_vec = vectorizer.fit_transform(X_train)

X_test_vec = vectorizer.transform(X_test)


from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

lr.fit(X_train_vec, y_train)

lr_score = lr.score(X_test_vec, y_test)
print("Results for Logistic Regression with tfidf")
print(lr_score)

y_pred_lr = lr.predict(X_test_vec)

#Confusion matrix
from sklearn.metrics import confusion_matrix
cm_knn = confusion_matrix(y_test, y_pred_lr)

tn, fp, fn, tp = confusion_matrix(y_test, y_pred_lr).ravel()
print(tn, fp, fn, tp)


tpr_knn = round(tp/(tp + fn), 4)
tnr_knn = round(tn/(tn+fp), 4)

print(tpr_knn, tnr_knn)




#Support Vector Machine
#from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                        test_size = 0.5, random_state=55)

vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

from sklearn import svm
#params = {'kernel':('linear', 'rbf'), 'C':[1, 10, 100]}
svcl = svm.SVC(kernel = 'rbf')
#clf_sv = GridSearchCV(svcl, params)
svcl.fit(X_train_vec, y_train)
svcl_score = svcl.score(X_test_vec, y_test)
print("Results for Support Vector Machine with tfidf")
print(svcl_score)

y_pred_sv = svcl.predict(X_test_vec)

#Confusion matrix
from sklearn.metrics import confusion_matrix
cm_sv = confusion_matrix(y_test, y_pred_sv)

tn, fp, fn, tp = confusion_matrix(y_test, y_pred_sv).ravel()
print(tn, fp, fn, tp)

tpr_sv = round(tp/(tp + fn), 4)
tnr_sv = round(tn/(tn+fp), 4)

print(tpr_sv, tnr_sv)




#KNN Classifier
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                        test_size = 0.5, random_state=65)

vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_vec, y_train)

knn_score = knn.score(X_test_vec, y_test)
print("Results for KNN Classifier with tfidf")
print(knn_score)


y_pred_knn = knn.predict(X_test_vec)

#Confusion matrix
cm_knn = confusion_matrix(y_test, y_pred_knn)

tn, fp, fn, tp = confusion_matrix(y_test, y_pred_knn).ravel()
print(tn, fp, fn, tp)


tpr_knn = round(tp/(tp + fn), 4)
tnr_knn = round(tn/(tn+fp), 4)

print(tpr_knn, tnr_knn)

