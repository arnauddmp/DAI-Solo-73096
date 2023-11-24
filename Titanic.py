# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 13:00:25 2023

@author: Arnaud de Dampierre
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

df_train=pd.read_csv('train.csv')   
df_test=pd.read_csv('test.csv') 

print(df_train.head(6))

df_test['Relatives']=df_test['SibSp']+df_test['Parch']
df_train['Relatives']=df_train['SibSp']+df_train['Parch']

plt.figure(figsize=(8, 6))
sns.countplot(x='Relatives', hue='Survived', data=df_train, palette='Set2')
plt.legend(title='Survived', labels=['Yes', 'No'])
plt.xlabel('Number of Relatives')
plt.show()


sns.countplot(x='Pclass', hue='Survived', data=df_train, palette='Set2')
plt.title('Rate of survival by class')
plt.xlabel('Class')
plt.ylabel('Count')
plt.legend(title='Survived', labels=['Yes', 'No'])
plt.show()

age_frame = [0, 18, 60, 80]
df_train['Age_Category'] = pd.cut(df_train['Age'], bins=age_frame, right=False)
sns.countplot(x='Age_Category', hue='Survived', data=df_train, palette='Set2')
plt.title('Survivability depending on the age')
plt.xlabel('Age')
plt.ylabel('Number of passengers')
plt.legend(title='Survived', labels=['No', 'Yes'])
plt.show()

sns.countplot(x='Sex', hue='Survived', data=df_train, palette='Set2')
plt.title('Survivors by gender')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.legend(title='Survived', labels=['No', 'Yes'])
plt.show()

X = df_train.drop(['Survived'], axis=1)
y = df_train['Survived']
X_second = pd.get_dummies(X)
X_third = X_second.fillna(X_second.mean())
X_train, X_test, y_train, y_test = train_test_split(X_third, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
lr_classifier = LogisticRegression(C=1)
lr_classifier.fit(X_train_scaled, y_train)
prediction = lr_classifier.predict(X_test_scaled)
accuracy = accuracy_score(y_test, prediction)
print(f'Accuracy: {accuracy}')

conf_matrix = confusion_matrix(y_test, prediction)
print("Confusion Matrix:")
print(conf_matrix)
