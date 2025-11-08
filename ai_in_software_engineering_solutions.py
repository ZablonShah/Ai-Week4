"""
File: ai_in_software_engineering_solutions.py
Author:Zablon Nyandika
Course: AI in Software Engineering - Week 4
Theme: Building Intelligent Software Solutions 💻🤖

This script contains three parts:
1. AI-Powered Code Completion
2. Automated Testing with AI (using Selenium)
3. Predictive Analytics for Resource Allocation

Dependencies:
    pip install pandas scikit-learn selenium
"""

# -----------------------------
# TASK 1: AI-POWERED CODE COMPLETION
# -----------------------------

def sort_dicts(data, key):
    """
    Sorts a list of dictionaries by a specific key.
    Example:
        data = [{'name':'Alice','age':25}, {'name':'Bob','age':22}]
        sort_dicts(data, 'age') -> [{'name':'Bob','age':22}, {'name':'Alice','age':25}]
    """
    if not data or key not in data[0]:
        print("Warning: Key not found or empty data list.")
        return data
    return sorted(data, key=lambda item: item[key])


# Demo run for Task 1
if __name__ == "__main__":
    sample_data = [
        {'name': 'Alice', 'age': 25},
        {'name': 'Bob', 'age': 22},
        {'name': 'Charlie', 'age': 30}
    ]
    print("=== Task 1: AI-Powered Code Completion ===")
    print("Original Data:", sample_data)
    sorted_data = sort_dicts(sample_data, 'age')
    print("Sorted Data by 'age':", sorted_data)


# -----------------------------
# TASK 2: AUTOMATED TESTING WITH AI
# -----------------------------
"""
Note: To run this section, ensure ChromeDriver is installed and accessible.
You can download it from https://chromedriver.chromium.org/downloads
"""

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import time

def automated_login_test():
    """
    Automates login test using Selenium.
    Replace URL and element IDs with actual values from your project.
    """
    print("\n=== Task 2: Automated Testing with AI ===")

    # Initialize WebDriver
    driver = webdriver.Chrome()
    driver.get("https://example-login-page.com")

    # Simulate user actions
    driver.find_element(By.ID, "username").send_keys("test_user")
    driver.find_element(By.ID, "password").send_keys("wrong_password")
    driver.find_element(By.ID, "login").click()

    # Wait for response
    time.sleep(2)

    # Validate login result
    if "Invalid" in driver.page_source:
        print("Test Passed: Invalid credentials correctly rejected.")
    else:
        print("Test Failed: Invalid credentials not detected.")

    driver.quit()

# Uncomment to run (requires live webpage)
# automated_login_test()


# -----------------------------
# TASK 3: PREDICTIVE ANALYTICS FOR RESOURCE ALLOCATION
# -----------------------------

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

def predictive_model():
    """
    Trains a RandomForestClassifier using the Kaggle Breast Cancer dataset
    and evaluates model performance.
    """
    print("\n=== Task 3: Predictive Analytics for Resource Allocation ===")

    # Load dataset
    # Download from: https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data
    data = pd.read_csv("breast_cancer_data.csv")

    # Basic preprocessing
    data = data.dropna()
    X = data.drop(['diagnosis'], axis=1)
    y = data['diagnosis'].map({'M': 1, 'B': 0})  # Encode labels

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Evaluate
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Accuracy: {acc:.3f}")
    print(f"F1 Score: {f1:.3f}")
