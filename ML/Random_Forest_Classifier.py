import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

target_col = ''

# Assume 'df' is your pandas DataFrame and 'target_col' is the name of your target column
X = df.drop(target_col, axis=1)
y = df[target_col]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the parameter grid for RandomizedSearchCV
param_grid = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}

# Initialize RandomForestClassifier
rf = RandomForestClassifier(random_state=42)

# Perform RandomizedSearchCV
random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_grid, 
                                   n_iter=100, cv=5, random_state=42, n_jobs=-1)
random_search.fit(X_train, y_train)

# Get the best parameters and model
best_params = random_search.best_params_
best_model = random_search.best_estimator_

# Evaluate the model
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Best parameters: {best_params}")
print(f"Accuracy: {accuracy}")

# Create the RFModels folder if it doesn't exist
if not os.path.exists('RFModels'):
    os.makedirs('RFModels')

# Save the best parameters and model
joblib.dump(best_params, 'RFModels/best_params.joblib')
joblib.dump(best_model, 'RFModels/best_model.joblib')

print("Best parameters and model saved in the RFModels folder.")
