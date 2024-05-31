import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Data loading and preprocessing
data = pd.read_csv('result.csv')
X = data.drop('num', axis=1)
y = data['num']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# XGBoost model creation and hyperparameter tuning for xg
param_grid_xg = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9]
}

grid_search_xg = GridSearchCV(estimator=XGBClassifier(random_state=42), param_grid=param_grid_xg, cv=5, n_jobs=1, verbose=2)
grid_search_xg.fit(X_train, y_train)

print(grid_search_xg.best_params_)

# Prediction and evaluation with optimal model for xg
best_model_xg = grid_search_xg.best_estimator_
y_pred_xg = best_model_xg.predict(X_test)


# Merge y_test and y_pred_xg
results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_xg})

# Calculate the frequency of each point
counts = results.groupby(['Actual', 'Predicted']).size().reset_index(name='counts')

# Draw bubble chart
plt.figure(figsize=(10, 6))
scatter = plt.scatter(counts['Actual'], counts['Predicted'], s=counts['counts']*100, c=counts['counts'], alpha=0.6, edgecolors='w', cmap='viridis')

# Add graph title and axis labels
plt.title('Bubble Chart of Actual vs Predicted')
plt.xlabel('Actual Value')
plt.ylabel('Predicted Value')

# Add legend for bubble sizes
for i in range(len(counts)):
    plt.text(counts['Actual'][i], counts['Predicted'][i], counts['counts'][i], ha='center', va='center', color='black')

plt.colorbar(scatter)
plt.show()

# Print evaluation results
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_xg))
print("Accuracy:", accuracy_score(y_test, y_pred_xg))
print("\nClassification Report:\n", classification_report(y_test, y_pred_xg))

# Visualize variable importance
ft_importance_xg = best_model_xg.feature_importances_

sorted_indices_xg = np.argsort(ft_importance_xg)[::-1]
ft_names = X.columns

plt.figure(figsize=(10, 6))
plt.bar(range(X.shape[1]), ft_importance_xg[sorted_indices_xg], align='center')
plt.xticks(range(X.shape[1]), ft_names[sorted_indices_xg], rotation=90)
plt.xlabel('Feature')
plt.ylabel('Feature Importance for xg')
plt.title('XGBoost Feature Importance')
plt.tight_layout()
plt.show()
