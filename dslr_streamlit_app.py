import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

train_data = pd.read_csv(r'D:\ain.csv')
test_data = pd.read_csv(r'D:\EXCEL R\Data Science\Assignments\6. Logistic Regression\Logistic Regression\Titanic_test.csv')
bank_data = pd.read_csv(r'D:\Assignment ExcelR\Logistic Regression\bank-full.csv', delimiter=';')
bank_data.head()

# Display the first few rows of the dataset
print(train_data.head())

# Summary statistics
print(train_data.describe())

# Data types of features
print(train_data.dtypes)

# Check for missing values
print(train_data.isnull().sum())

# Visualizations
# 1. Distribution of Age
sns.histplot(train_data['Age'].dropna(), kde=True)
plt.title('Age Distribution')
plt.show()

# 2. Boxplot of Age vs Survival
sns.boxplot(x='Survived', y='Age', data=train_data)
plt.title('Age vs Survival')
plt.show()

# 3. Pair plot of a few features
sns.pairplot(train_data[['Survived', 'Age', 'Fare', 'Pclass']], hue='Survived')
plt.show()

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Handling missing values: Impute missing Age with the median
imputer = SimpleImputer(strategy='median')
train_data['Age'] = imputer.fit_transform(train_data[['Age']])

label_encoder = LabelEncoder()
train_data['Sex'] = label_encoder.fit_transform(train_data['Sex'])
train_data['Embarked'] = label_encoder.fit_transform(train_data['Embarked'].fillna('S'))

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Features and target variable
X = train_data[['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']]
y = train_data['Survived']


from sklearn.linear_model import LogisticRegression

# Create the Logistic Regression model
model = LogisticRegression()

# Train the model on the training data
model.fit(X_train, y_train)


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve

# Predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)

# Print evaluation metrics
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-Score: {f1}")
print(f"ROC-AUC Score: {roc_auc}")


# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:,1])

plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

# Displaying the coefficients of the logistic regression model
coefficients = pd.DataFrame(model.coef_, columns=X.columns)
print("Logistic Regression Coefficients:")
print(coefficients)

# Interpreting the coefficients
for feature, coef in zip(X.columns, model.coef_[0]):
    print(f"Feature: {feature}, Coefficient: {coef:.4f}")

pip install streamlit

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

# Load the model and necessary data
model = LogisticRegression()
train_data = pd.read_csv(r'D:\EXCEL R\Data Science\Assignments\6. Logistic Regression\Logistic Regression\Titanic_train.csv')

# Preprocessing (same as before)
imputer = SimpleImputer(strategy='median')
train_data['Age'] = imputer.fit_transform(train_data[['Age']])
label_encoder = LabelEncoder()
train_data['Sex'] = label_encoder.fit_transform(train_data['Sex'])
train_data['Embarked'] = label_encoder.fit_transform(train_data['Embarked'].fillna('S'))

# Features for prediction
X = train_data[['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']]
y = train_data['Survived']
model.fit(X, y)

# Streamlit app
st.title('Titanic Survival Prediction')
Pclass = st.selectbox('Class', [1, 2, 3])
Sex = st.selectbox('Sex', ['male', 'female'])
Age = st.number_input('Age', min_value=0, max_value=100, value=30)
Fare = st.number_input('Fare', min_value=0, value=7)
Embarked = st.selectbox('Embarked', ['C', 'Q', 'S'])

# Preprocess the input data
input_data = pd.DataFrame([[Pclass, Sex, Age, Fare, Embarked]], columns=['Pclass', 'Sex', 'Age', 'Fare', 'Embarked'])
input_data['Sex'] = label_encoder.transform(input_data['Sex'])
input_data['Embarked'] = label_encoder.transform(input_data['Embarked'])

# Make prediction
prediction = model.predict(input_data)

# Display result
if prediction == 1:
    st.write("Survival Prediction: Survived!")
else:
    st.write("Survival Prediction: Did not survive.")
