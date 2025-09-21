import pandas as pd
import numpy as np

df = pd.read_csv('diabetes (2).csv')

print(df)

df



df.info()

df.describe()

df.duplicated().sum()

df.isnull().sum()

Separate dependent and independent variables

Dependent: Outcome

Independent: All others

X = df.drop(columns=['Outcome'])
y = df['Outcome']

df['Outcome']

df['Outcome'].value_counts().plot(kind='pie')

df['Outcome'].value_counts().plot(kind='bar')

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X.shape)
print(X_train.shape)
print(X_test.shape)

from sklearn.linear_model import LogisticRegression

logReg = LogisticRegression()

logReg.fit(X_train,y_train)

y_pred = logReg.predict(X_test)

y_pred

from sklearn.metrics import confusion_matrix, classification_report
print("The confusion matrix is:\n")
print(confusion_matrix(y_test,y_pred))
print("\nThe classification report is:\n")
print(classification_report(y_test,y_pred))

