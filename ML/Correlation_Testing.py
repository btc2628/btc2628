import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score



df = pd.DataFrame(None)

# Define the target column
target_column = 'Target'

# Compute correlation matrix
correlation_matrix = df.corr()

# Get correlation of all columns with the target column
target_correlations = correlation_matrix[target_column].drop(target_column)

# Identify strongly correlated and uncorrelated features
strongly_correlated = target_correlations[target_correlations.abs() > 0.5]
uncorrelated = target_correlations[target_correlations.abs() <= 0.1]

print("Strongly Correlated Features with Target:\n", strongly_correlated)
print("\nUncorrelated Features with Target:\n", uncorrelated)

# Prepare the data for regression
X = df.drop(columns=[target_column])
y = df[target_column]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate R-squared (coefficient of determination)
r_squared = r2_score(y_test, y_pred)

print("\nR-squared value (how much of the target is explained by other features):", r_squared)
