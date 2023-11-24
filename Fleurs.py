# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 12:57:47 2023

@author: Arnaud de Dampierre
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

iris = pd.read_csv("Fleurs.csv")
sns.pairplot(iris,hue="species",height=3);
plt.show()

sns.boxplot(x="species",y="petal_length",data=iris)
plt.show()

x = iris.drop('species', axis=1)
y = iris['species']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
svm = SVC(C=1)
svm.fit(x_train, y_train)
prediction = svm.predict(x_test)
accuracy = accuracy_score(y_test, prediction)
print("Accuracy is up to {}".format(accuracy))

matrix = confusion_matrix(y_test, prediction)
print("Confusion Matrix:")
print(matrix)