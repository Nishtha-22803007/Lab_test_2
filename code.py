import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score
housing_data=fetch_california_housing()
housing_data
df = pd.DataFrame(housing_data.data, columns=housing_data.feature_names)
df['target'] = housing_data.target   
df.head()
df.info()
feature_types = df.dtypes
num_features = feature_types[feature_types == 'float64']
num_features_count=num_features.count()
cat_features = feature_types[feature_types == 'object']
cat_features_count=cat_features.count()

print('Numerical features-')
print(num_features)
print('\nNumber of numerical features: ',num_features_count)
print('Categorical features-')
print(cat_features)
print('\nNumber of categorical features: ',cat_features_count)
plt.figure(figsize=(10, 6))
plt.plot(df['AveBedrms'], df['target'], 'o', alpha=0.5)
plt.xlabel('Average Number of BedRooms')
plt.ylabel('House Price (Target)')
plt.title('House Price vs. Average Number of BedRooms')
plt.grid(True)
plt.show()
df.isnull().sum()
X=housing_data.data
y=housing_data.target
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.4,random_state=42)
model_linear=LinearRegression()
model_linear.fit(X_train,y_train)
model_svm=SVR() 
model_svm.fit(X_train,y_train)
linear_pred=model_linear.predict(X_test)
svm_pred=model_svm.predict(X_test)
linear_pred
svm_pred
y_test_bins = pd.cut(y_test, bins=5, labels=[0, 1, 2, 3,4])

lr_pred_bins = pd.cut(linear_pred, bins=5, labels=[0, 1, 2, 3, 4])
svm_pred_bins = pd.cut(svm_pred, bins=5, labels=[0, 1, 2, 3, 4])

lr_accuracy = accuracy_score(y_test_bins, lr_pred_bins)
svm_accuracy = accuracy_score(y_test_bins, svm_pred_bins)

lr_precision = precision_score(y_test_bins, lr_pred_bins, average='weighted')
svm_precision = precision_score(y_test_bins, svm_pred_bins, average='weighted')

lr_recall = recall_score(y_test_bins, lr_pred_bins, average='weighted')
svm_recall = recall_score(y_test_bins, svm_pred_bins, average='weighted')

lr_f1 = f1_score(y_test_bins, lr_pred_bins, average='weighted')
svm_f1 = f1_score(y_test_bins, svm_pred_bins, average='weighted')

print(f'\nLinear Regression Accuracy: {lr_accuracy}')
print(f'SVM Accuracy: {svm_accuracy}')
print(f'Linear Regression Precision: {lr_precision}')
print(f'SVM Precision: {svm_precision}')
print(f'Linear Regression Recall: {lr_recall}')
print(f'SVM Recall: {svm_recall}')
print(f'Linear Regression F1 Score: {lr_f1}')
print(f'SVM F1 Score: {svm_f1}')

