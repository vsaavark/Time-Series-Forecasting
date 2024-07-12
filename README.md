# Time-Series-Forecasting

# Overview
This project aims to forecast hourly energy consumption using machine learning techniques. By analyzing historical energy consumption data, we can predict future consumption patterns and assist in optimizing energy usage and management.

# Dataset
The dataset for this project is imported from Kaggle, titled "Hourly Energy Consumption" by Rob Mulla. It contains hourly energy consumption data that serves as the basis for building and validating the forecasting model.

# Tools and Libraries Used
pandas: For data manipulation
matplotlib: For data visualization
scikit-learn: For machine learning model creation and evaluation
numpy: For numerical operations
seaborn: For statistical data visualization
xgboost: For implementing the XGBoost algorithm
Installation
To install the necessary libraries, you can use pip:

pip install pandas matplotlib scikit-learn numpy seaborn xgboost

# Steps and Methodology
Data Import and Preprocessing

Load the dataset and perform initial data cleaning and preprocessing.
Train/Test Split

Split the dataset into training and testing sets to evaluate model performance.
Feature Creation

Generate relevant features from the time series data to enhance model accuracy.
Features include date and time components such as hour, day of the week, and month.
Visualize Features/Target Relationships

Use visualization techniques to understand the relationships between features and the target variable (energy consumption).
Model Creation

Implement machine learning models to forecast energy consumption.
Experiment with different algorithms and select the best-performing model.
Feature Importance

Analyze the importance of different features in predicting energy consumption.
Forecast on Test Data

Use the trained model to forecast energy consumption on the test dataset.
Evaluate Model Performance

Calculate the Root Mean Squared Error (RMSE) to assess the accuracy of the forecast.
Analyze forecast errors to identify areas for improvement.
# Example Code

python

Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import mean_squared_error

Load Dataset
data = pd.read_csv('hourly_energy_consumption.csv')

Feature Creation
data['hour'] = data['datetime'].dt.hour
data['day_of_week'] = data['datetime'].dt.dayofweek
data['month'] = data['datetime'].dt.month

Train/Test Split
train, test = train_test_split(data, test_size=0.2, shuffle=False)

Model Creation
X_train = train[['hour', 'day_of_week', 'month']]
y_train = train['energy_consumption']
X_test = test[['hour', 'day_of_week', 'month']]
y_test = test['energy_consumption']

model = xgb.XGBRegressor()
model.fit(X_train, y_train)

Forecast on Test Data
y_pred = model.predict(X_test)

Evaluate Model Performance
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'RMSE: {rmse}')

Feature Importance
xgb.plot_importance(model)
plt.show()

# Conclusion
This project demonstrates the application of machine learning techniques to forecast hourly energy consumption. The use of feature engineering and model evaluation ensures accurate and reliable predictions, aiding in efficient energy management.
