# Product Purchase Prediction Using Machine Learning Algorithms

This project applies three machine learning algorithms—**Logistic Regression**, **K-Nearest Neighbors (KNN)**, and **Gaussian Naive Bayes**—to classify whether a user has purchased a product based on features such as **age**, **gender**, and **salary**. The dataset used contains user information, and the target variable is whether the user has purchased the product (`Purchased`).

## Overview
The goal of this project is to predict whether a user will purchase a product based on their **age**, **gender**, and **estimated salary** using three classification algorithms:
- **Logistic Regression**
- **K-Nearest Neighbors (KNN)**
- **Gaussian Naive Bayes**

We compare the performance of these algorithms using metrics like accuracy, precision, recall, F1 score, and ROC-AUC.

## Dataset
The dataset contains the following features:
- **User ID**: A unique identifier for each user (removed during preprocessing).
- **Gender**: The gender of the user (converted to numeric values).
- **Age**: The age of the user.
- **Estimated Salary**: The estimated salary of the user.
- **Purchased**: The target variable (0 = No, 1 = Yes), indicating whether the user purchased the product.

### Data Preprocessing
- Gender has been converted to numerical values using one-hot encoding.
- Unnecessary columns (e.g., `User ID`) have been dropped.
- The dataset has been split into training and testing sets.

## Algorithms Used
### 1. Logistic Regression
Logistic Regression is a linear model used to predict the probability of a binary outcome. In this project, it is used to model the relationship between the user's age, gender, salary, and their likelihood of purchasing the product.

### 2. K-Nearest Neighbors (KNN)
KNN is a non-parametric method that classifies a data point based on how its neighbors are classified. It is intuitive and works well for this dataset with a small number of features.

### 3. Gaussian Naive Bayes
Gaussian Naive Bayes is a probabilistic algorithm based on Bayes' Theorem. It assumes that the features follow a Gaussian (normal) distribution. This algorithm is especially useful when the features are continuous, as in the case of age and salary.
