## Reading Dataset
import pandas as pd
dataset = pd.read_csv('Fraud.csv')

## Dropping Unneccesary Features
dataset = dataset.drop(columns=['nameOrig','nameDest'])

## Checking for missing values
# print(dataset.isnull().sum())

## Checking for outliers
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

sns.boxplot(x=dataset['amount'])
# plt.show()

Q1 = dataset['amount'].quantile(0.25)
Q3 = dataset['amount'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

dataset['log_amount'] = np.log1p(dataset['amount'])

upper_cap = dataset['amount'].quantile(0.99)
lower_cap = dataset['amount'].quantile(0.01)
dataset['amount_capped'] = np.where(dataset['amount'] > upper_cap, upper_cap,
                        np.where(dataset['amount'] < lower_cap, lower_cap, dataset['amount']))

dataset = dataset[dataset['amount'] >= 0]

dataset['is_outlier_amount'] = ((dataset['amount'] < lower_bound) | (dataset['amount'] > upper_bound)).astype(int)

# dataset['type'] = dataset['type'].map({'CASH-IN':1, 'CASH-OUT':2,'DEBIT':3,'PAYMENT':4,'TRANSFER':5})

## Checking for multi-collinearity
# corr_matrix = dataset.corr()
# plt.figure(figsize=(10,8))
# sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
# plt.show()

## Dropping Features due to multi-collinearity
dataset = dataset.drop(columns=['oldbalanceOrg','oldbalanceDest'])

# dataset['type'] = dataset['type'].map({1:'CASH-IN',2:'CASH-OUT',3:'DEBIT',4:'PAYMENT',5:'TRANSFER'})

X = dataset.drop(columns=['isFraud'])
y = dataset['isFraud']

## Spliting data for train and test
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=28)

## Pipelining
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

num_features = ['step','amount','newbalanceOrig','newbalanceDest','isFlaggedFraud','log_amount','amount_capped', 'is_outlier_amount']
cat_features = ['type']

categorical_pipeline = Pipeline([
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

numeric_pipeline = Pipeline([
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer([
    ('num', numeric_pipeline, num_features),
    ('cat', categorical_pipeline, cat_features)
])

## Using SMOTE for imbalance
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=28)
X_train_preprocessed = preprocessor.fit_transform(X_train)
X_train_res, y_train_res = smote.fit_resample(X_train_preprocessed, y_train)

model = LogisticRegression()
model.fit(X_train_res,y_train_res)

X_test_preprocessed = preprocessor.transform(X_test)

y_pred = model.predict(X_test_preprocessed)

## Evaluating Model
from sklearn.metrics import precision_score,recall_score,f1_score
print('Precision:',precision_score(y_test,y_pred))
print('Recall:',recall_score(y_test,y_pred))
print('F1-score:',f1_score(y_test,y_pred))

## Results:

# With Logistics Regression + SMOTE
# Precision: 0.005999640593347709
# Recall: 0.8785885167464115
# F1-score: 0.011917897128022067