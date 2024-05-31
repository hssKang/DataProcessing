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


# XGBoost model creation and hyperparameter tuning
param_grid_xg = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9]
}

# Perform grid search for various k values
k_values = [3, 5, 7, 10]
best_scores = []
best_params = []
best_models = []

for k in k_values:
    grid_search_xg = GridSearchCV(estimator=XGBClassifier(random_state=42), param_grid=param_grid_xg, cv=k, n_jobs=1, verbose=2)
    grid_search_xg.fit(X_train, y_train)
    best_scores.append(grid_search_xg.best_score_)
    best_params.append(grid_search_xg.best_params_)
    best_models.append(grid_search_xg.best_estimator_)

# Find the optimal k value
best_k_index = np.argmax(best_scores)
best_k = k_values[best_k_index]
best_model_xg = best_models[best_k_index]


print(f"Best k value: {best_k}")
print(f"Best parameters for k={best_k}: {best_params[best_k_index]}")

# Prediction and evaluation with optimal model
y_pred_xg = best_model_xg.predict(X_test)

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
plt.title('various k-value XGBoost Feature Importance')
plt.tight_layout()
plt.show()
