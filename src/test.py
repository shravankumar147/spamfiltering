import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score

# read data file using pandas csv reader
df = pd.read_csv('../dataset/SMSSpamCollection', delimiter='\t', header=None)

# print out the first 5 SMS
print("sample instances: ")
print(df.head())

# print 'Number of spam messages:', df[df[0] == 'spam'][0].count()
# print 'Number of ham messages:', df[df[0] == 'ham'][0].count()

# pre processing the data
X_train_raw, X_test_raw, y_train, y_test = train_test_split(df[1],df[0])
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train_raw)
X_test = vectorizer.transform(X_test_raw)

# model creation and training
classifier = LogisticRegression()
classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)

print("Accuracy on test data:")

score =classifier.score(X_test, y_test)
print("Accuracy: {}".format(score))

print("Cross Validation Accuracy:")
scores = cross_val_score(classifier, X_test, y_test, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

