import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy.stats import uniform, randint

# Load the dataset
df = pd.read_csv(r"C:\Users\mahno\OneDrive\Desktop\internship\emails.csv")

# Extract features (X) and target variable (y)
X = df.iloc[:, 1:3001]  # Features
y = df['Prediction']    # Target variable

# Scale the features to [0, 1] range
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Use a smaller subset of data for quick experimentation
X_subset = X_scaled[:1000]
y_subset = y[:1000]

# Split data into training and testing sets
train_x, test_x, train_y, test_y = train_test_split(X_subset, y_subset, test_size=0.25, random_state=42)

# Define hyperparameter grids for each model
param_grids = {
    "MultinomialNB": {
        "alpha": [0.1, 0.5, 1.0, 1.5, 2.0]
    },
    "SVC": {
        "C": [0.1, 1, 10, 100],
        "gamma": ['scale', 'auto'],
        "kernel": ['rbf', 'linear']
    },
    "RandomForest": {
        "n_estimators": [10, 50, 100],
        "criterion": ['gini', 'entropy'],
        "max_features": ['sqrt', 'log2', None]
    },
    "LogisticRegression": {
        "C": uniform(0.1, 10),
        "solver": ['newton-cg', 'lbfgs', 'liblinear']
    },
    "DecisionTree": {
        "criterion": ['gini', 'entropy'],
        "max_depth": randint(1, 20),
        "min_samples_split": randint(2, 10)
    }
}

# Initialize models
models = {
    "MultinomialNB": MultinomialNB(),
    "SVC": SVC(),
    "RandomForest": RandomForestClassifier(),
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "DecisionTree": DecisionTreeClassifier()
}

# Function to perform hyperparameter tuning using GridSearchCV
def grid_search_cv(model, param_grid):
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(train_x, train_y)
    return grid_search.best_estimator_

# Function to perform hyperparameter tuning using RandomizedSearchCV
def random_search_cv(model, param_distributions, n_iter=100):
    random_search = RandomizedSearchCV(model, param_distributions, n_iter=n_iter, cv=5, scoring='accuracy', n_jobs=-1, random_state=42)
    random_search.fit(train_x, train_y)
    return random_search.best_estimator_

# Perform hyperparameter tuning for each model
tuned_models = {}
for name, model in models.items():
    if name in param_grids:
        if name == "LogisticRegression" or name == "DecisionTree":
            print(f"\nPerforming Randomized Search for {name}...")
            tuned_model = random_search_cv(model, param_grids[name])
        else:
            print(f"\nPerforming Grid Search for {name}...")
            tuned_model = grid_search_cv(model, param_grids[name])
        tuned_models[name] = tuned_model
        print(f"Best parameters for {name}: {tuned_model}")

# Evaluate the performance of tuned models
for name, model in tuned_models.items():
    y_pred = model.predict(test_x)
    accuracy = accuracy_score(test_y, y_pred)
    precision = precision_score(test_y, y_pred, average='macro', zero_division=0)
    recall = recall_score(test_y, y_pred, average='macro', zero_division=0)
    f1 = f1_score(test_y, y_pred, average='macro', zero_division=0)
    
    print(f"\n{name} Performance after Hyperparameter Tuning:")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
