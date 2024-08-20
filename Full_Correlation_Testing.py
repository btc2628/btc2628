import pandas as pd
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from dcor import distance_correlation
import matplotlib.pyplot as plt
from pygam import LinearGAM

# Assume df is already defined
# Example: df = pd.read_csv('your_dataset.csv')

# Define the target column
target_column = 'Target'

# Iterate over all features
for feature in df.columns:
    if feature != target_column:
        print(f"Analyzing feature: {feature}\n")

        # Pearson Correlation
        pearson_corr, pearson_p_value = pearsonr(df[feature], df[target_column])
        print(f"Pearson correlation: {pearson_corr}, p-value: {pearson_p_value}")

        # Spearman's Rank Correlation
        spearman_corr, spearman_p_value = spearmanr(df[feature], df[target_column])
        print(f"Spearman's rank correlation: {spearman_corr}, p-value: {spearman_p_value}")

        # Kendall's Tau
        kendall_tau, kendall_p_value = kendalltau(df[feature], df[target_column])
        print(f"Kendall's Tau: {kendall_tau}, p-value: {kendall_p_value}")

        # Mutual Information
        mi = mutual_info_regression(df[[feature]], df[target_column])
        print(f"Mutual Information: {mi[0]}")

        # Distance Correlation
        dcor_value = distance_correlation(df[feature], df[target_column])
        print(f"Distance Correlation: {dcor_value}")

        # Polynomial Regression for Non-linear Relationship
        poly = PolynomialFeatures(degree=2)
        X_poly = poly.fit_transform(df[[feature]])

        model = LinearRegression()
        model.fit(X_poly, df[target_column])
        y_pred = model.predict(X_poly)

        r_squared = r2_score(df[target_column], y_pred)
        print(f"R-squared for Polynomial Regression (degree 2): {r_squared}")

        # Generalized Additive Model (GAM)
        gam = LinearGAM().fit(df[[feature]], df[target_column])
        gam.summary()

        # Scatter Plot Visualization
        plt.scatter(df[feature], df[target_column])
        plt.xlabel(feature)
        plt.ylabel(target_column)
        plt.title(f'Scatter plot between {feature} and {target_column}')
        plt.show()

        print("\n" + "-"*50 + "\n")
