#Program to learn ML using Python using scikit-learn

import numpy as np
from sklearn import datasets, preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_squared_error
import matplotlib.pyplot as plt

def load_dataset(choice):
    if choice == 1:
        return datasets.load_iris()
    elif choice == 2:
        return datasets.load_boston()
    elif choice == 3:
        return datasets.load_digits()
    else:
        return None

def linear_regression_demo(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f"\nMean Squared Error: {mse:.2f}")
    print(f"Model Score: {model.score(X_test, y_test):.2f}")

def logistic_regression_demo(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"\nAccuracy Score: {accuracy:.2f}")
    print(f"Model Score: {model.score(X_test, y_test):.2f}")

def decision_tree_demo(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"\nAccuracy Score: {accuracy:.2f}")
    print(f"Model Score: {model.score(X_test, y_test):.2f}")

def random_forest_demo(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"\nAccuracy Score: {accuracy:.2f}")
    print(f"Model Score: {model.score(X_test, y_test):.2f}")

def main():
    while True:
        print("\n=== ML Learning Program using Scikit-learn ===")
        print("1. Load Iris Dataset (Classification)")
        print("2. Load Boston Housing Dataset (Regression)")
        print("3. Load Digits Dataset (Classification)")
        print("4. Exit")
        
        dataset_choice = int(input("\nSelect dataset (1-4): "))
        
        if dataset_choice == 4:
            print("Thank you for learning ML!")
            break
            
        dataset = load_dataset(dataset_choice)
        if dataset is None:
            print("Invalid choice!")
            continue
            
        X = dataset.data
        y = dataset.target
        
        print("\nSelect ML Algorithm:")
        print("1. Linear Regression")
        print("2. Logistic Regression")
        print("3. Decision Tree")
        print("4. Random Forest")
        
        algo_choice = int(input("\nSelect algorithm (1-4): "))
        
        if algo_choice == 1:
            linear_regression_demo(X, y)
        elif algo_choice == 2:
            logistic_regression_demo(X, y)
        elif algo_choice == 3:
            decision_tree_demo(X, y)
        elif algo_choice == 4:
            random_forest_demo(X, y)
        else:
            print("Invalid algorithm choice!")

if __name__ == "__main__":
    main()
