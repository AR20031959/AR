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
