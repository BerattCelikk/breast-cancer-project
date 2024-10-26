# Breast Cancer Diagnosis Using Decision Tree Classifier 🌸

## Overview
This project implements a Decision Tree Classifier to diagnose breast cancer based on features extracted from a dataset. The dataset contains various measurements and characteristics, allowing the model to classify tumors as either malignant or benign.

## Libraries Used 📚
The project utilizes several Python libraries for data manipulation, visualization, and machine learning:

- **Pandas**: For data manipulation and analysis.
- **NumPy**: For numerical operations.
- **Matplotlib**: For basic data visualization.
- **Seaborn**: For advanced data visualization.
- **Scikit-learn**: For building and evaluating the Decision Tree model, including functions for hyperparameter tuning and performance metrics.

## Dataset
The dataset used in this project is a CSV file named `breast-cancer.csv`, which contains features related to breast cancer tumors along with a diagnosis label (Malignant or Benign).

## Code Explanation

### 1. Importing Libraries
```python
import pandas as pd  # For data manipulation and analysis
import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For data visualization
import seaborn as sns  # For advanced data visualization
from sklearn.tree import DecisionTreeClassifier, plot_tree  # Decision Tree Classifier and visualization
from sklearn.model_selection import train_test_split, GridSearchCV  # Dataset splitting and hyperparameter tuning
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix  # Model evaluation metrics
```
This section imports all necessary libraries for data analysis, visualization, and machine learning.

2. Loading the Dataset
```python
cancer = pd.read_csv("breast-cancer.csv")  # Reading the CSV file containing the dataset
```
The dataset is loaded into a Pandas DataFrame for further processing.

3. Data Overview
```python
print("First 3 rows of the dataset:\n", cancer.head(3))  # Printing the first three rows of the dataset
cancer.info()  # Providing details about the dataset structure
```
This part displays the first three rows and provides information about the dataset structure, including data types and missing values.

4. Data Preparation
```python
y = cancer["diagnosis"].map({'M': 1, 'B': 0})  # Encoding 'M' as 1 and 'B' as 0 for the target variable
x = cancer.drop(columns=["diagnosis", "id"], axis=1)  # Dropping non-feature columns for model training
```
The target variable (diagnosis) is encoded as binary values: 1 for Malignant and 0 for Benign. The non-feature columns are dropped to prepare the dataset for model training.

5. Splitting the Dataset
```python
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=42)
```
The dataset is split into training (80%) and testing (20%) sets to evaluate the model's performance later.

6. Training the Decision Tree Classifier
```python
tree = DecisionTreeClassifier(random_state=42)  # Initializing the Decision Tree Classifier
param_grid = {
    'max_depth': [3, 5, 7, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
grid_search = GridSearchCV(tree, param_grid, cv=5, scoring='accuracy', n_jobs=-1)  
grid_search.fit(x_train, y_train)  # Fit grid search to the training data
```
A Decision Tree Classifier is initialized, and hyperparameter tuning is performed using GridSearchCV to find the best parameters for the model.

7. Model Evaluation
```python
y_pred = best_tree.predict(x_test)  # Making predictions on the test set
accuracy = accuracy_score(y_test, y_pred)  # Calculating accuracy
print(f"\nModel Accuracy: {accuracy:.2f}")  # Displaying accuracy
print("\nClassification Report:\n", classification_report(y_test, y_pred))
```
The model's predictions are evaluated against the test set, and metrics such as accuracy and a classification report are generated.

8. Confusion Matrix Visualization
```python
conf_matrix = confusion_matrix(y_test, y_pred)  # Creating confusion matrix
plt.figure(figsize=(8, 6))  # Setting figure size
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Benign', 'Malignant'], yticklabels=['Benign', 'Malignant'])  
plt.title('Confusion Matrix')  # Title for the plot
plt.ylabel('Actual')  # Y-axis label
plt.xlabel('Predicted')  # X-axis label
plt.show()  # Display the plot
```
The confusion matrix is created and visualized using a heatmap for better interpretation of model performance.

9. Decision Tree Visualization
```python
plt.figure(figsize=(12, 8))  # Setting figure size for the tree plot
plot_tree(best_tree, feature_names=x.columns, class_names=['Benign', 'Malignant'], filled=True, rounded=True)  
plt.title('Decision Tree Visualization')  # Title for the plot
plt.show()  # Display the tree plot
```
The structure of the trained decision tree is visualized, allowing users to see the decision-making process.

10. Feature Importance Analysis
```python
feature_importance = best_tree.feature_importances_  # Getting feature importance scores
importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importance})  
importance_df = importance_df.sort_values(by='Importance', ascending=False)  # Sorting by importance scores
```
The importance of each feature in predicting the diagnosis is analyzed and displayed in a sorted DataFrame.

11. Making Predictions on New Data
```python
def make_prediction(input_data):
    ...
new_data = {
    "feature1": 10,  # Replace with actual feature name and value
    ...
}
predicted_diagnosis = make_prediction(new_data)  # Making a prediction for new data
```
A function is defined to make predictions on new data, allowing users to input their own feature values to receive a diagnosis prediction.

Conclusion
This project demonstrates how to use a Decision Tree Classifier for breast cancer diagnosis, providing insights into model training, evaluation, and visualization. The code is well-structured and can serve as a foundation for further exploration and improvement in machine learning applications.
