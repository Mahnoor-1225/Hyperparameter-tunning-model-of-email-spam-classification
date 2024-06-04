import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import randint

# Load the dataset
df = pd.read_csv(r"C:\Users\mahno\OneDrive\Desktop\internship\emails.csv")

# Extract features (X) and target variable (y)
X = df.iloc[:, 1:-1]  # All columns except the first and last ones
y = df.iloc[:, -1]    # Last column as target

# Scale the features to [0, 1] range
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
train_x, test_x, train_y, test_y = train_test_split(X_scaled, y, test_size=0.25, random_state=42)

# Define the RandomForestClassifier model
rf = RandomForestClassifier(random_state=42)

# Define hyperparameter grid for Grid Search
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'criterion': ['gini', 'entropy']
}

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)

# Perform Grid Search
grid_search.fit(train_x, train_y)

# Print the best parameters and the best score from Grid Search
print("Best Hyperparameters from Grid Search:", grid_search.best_params_)
print("Best Grid Search Score:", grid_search.best_score_)

# Define hyperparameter distribution for Random Search
param_dist = {
    'n_estimators': randint(50, 200),
    'max_depth': [None, 10, 20, 30],
    'criterion': ['gini', 'entropy']
}

# Initialize RandomizedSearchCV
random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_dist, n_iter=10, cv=5, scoring='accuracy', n_jobs=-1, random_state=42)

# Perform Random Search
random_search.fit(train_x, train_y)

# Print the best parameters and the best score from Random Search
print("Best Hyperparameters from Random Search:", random_search.best_params_)
print("Best Random Search Score:", random_search.best_score_)

# Evaluate the best model from Grid Search on the test data
best_rf_grid = grid_search.best_estimator_
y_pred_grid = best_rf_grid.predict(test_x)
print("\nGrid Search Model Evaluation:")
print("Accuracy:", accuracy_score(test_y, y_pred_grid))
print("Precision:", precision_score(test_y, y_pred_grid))
print("Recall:", recall_score(test_y, y_pred_grid))
print("F1 Score:", f1_score(test_y, y_pred_grid))

# Evaluate the best model from Random Search on the test data
best_rf_random = random_search.best_estimator_
y_pred_random = best_rf_random.predict(test_x)
print("\nRandom Search Model Evaluation:")
print("Accuracy:", accuracy_score(test_y, y_pred_random))
print("Precision:", precision_score(test_y, y_pred_random))
print("Recall:", recall_score(test_y, y_pred_random))
print("F1 Score:", f1_score(test_y, y_pred_random))
