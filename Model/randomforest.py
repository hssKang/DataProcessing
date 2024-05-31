import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Data loading and preprocessing
data = pd.read_csv('result.csv')
X = data.drop('num', axis=1)
y = data['num']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Random forest model creation and hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 10, 20],
    'min_samples_leaf': [1, 5, 10],
    'bootstrap': [True, False],
    'criterion': ['gini', 'entropy']
}

grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42), param_grid=param_grid, cv=5, n_jobs=1, verbose=2)
grid_search.fit(X_train, y_train)

# Optimal hyperparameter output
print(f"Best parameters: {grid_search.best_params_}")

# Prediction and evaluation with optimal model
best_rf_model = grid_search.best_estimator_
y_pred = best_rf_model.predict(X_test)

# Merge y_test and y_pred
results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

# Calculate the frequency of each point
counts = results.groupby(['Actual', 'Predicted']).size().reset_index(name='counts')

# Draw bubble chart
plt.figure(figsize=(10, 6))
scatter = plt.scatter(counts['Actual'], counts['Predicted'], s=counts['counts']*100, alpha=0.6, edgecolors='w')

# Add graph title and axis labels
plt.title('Bubble Chart of Actual vs Predicted')
plt.xlabel('Actual Value')
plt.ylabel('Predicted Value')

# Add legend for bubble sizes
for i in range(len(counts)):
    plt.text(counts['Actual'][i], counts['Predicted'][i], counts['counts'][i], ha='center', va='center', color='black')

plt.show()

# Print evaluation results
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Visualize variable importance
ft_importance_rf = best_rf_model.feature_importances_

sorted_indices_rf = np.argsort(ft_importance_rf)[::-1]
ft_names = X.columns

plt.figure(figsize=(10, 6))
plt.bar(range(X.shape[1]), ft_importance_rf[sorted_indices_rf], align='center')
plt.xticks(range(X.shape[1]), ft_names[sorted_indices_rf], rotation=90)
plt.xlabel('Feature')
plt.ylabel('Feature Importance')
plt.title('Random Forest Feature Importance')
plt.tight_layout()
plt.show()
