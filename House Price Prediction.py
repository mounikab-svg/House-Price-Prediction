# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Step 2: Load the Dataset from OpenML (replaces load_boston)
boston = fetch_openml(name="boston", version=1, as_frame=True)
df = boston.frame  # converts to pandas DataFrame

# Step 3: Rename Target Column
df.rename(columns={'MEDV': 'PRICE'}, inplace=True)

# Step 4: Display first few rows
print("✅ Dataset Loaded Successfully!")
print(df.head())

# Step 5: Visualize a Feature vs Target
sns.scatterplot(x='RM', y='PRICE', data=df)
plt.title('Average Number of Rooms vs House Price')
plt.show()

# Step 6: Split Data
X = df.drop('PRICE', axis=1)
y = df['PRICE']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 8: Train the Model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Step 9: Predict
y_pred = model.predict(X_test_scaled)

# Step 10: Evaluate Model
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# Step 11: Visualization - Actual vs Predicted
plt.scatter(y_test, y_pred, color='blue')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted House Prices')
plt.show()
