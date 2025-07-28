import numpy as np
import pandas as pd
import random
import script
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score

# features and data
file_path = 'sample-data.csv'
data = script.network_attack_preprocessing_pipeline(file_path)
X = data['X']
y = data['y']
X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state=42)


print("=" * 50)
print("\nStarting 5-fold Cross Validation for KNN...")

K_vals = list(range(5, 51, 5))
K_accuracy = []
np.random.seed(100)

# find the best value of K using 5-fold cross validation
CV_folds = np.random.choice([1, 2, 3, 4, 5], size=len(X_train), replace=True)
num_folds = 5

for i,k in enumerate(K_vals):
    K = k
    CV_accuracy = np.zeros(num_folds)

    for j in range(1, num_folds+1):
        cv_X_train = X_train[CV_folds != j]
        cv_y_train = y_train[CV_folds != j]
        cv_X_val = X_train[CV_folds == j]
        cv_y_val = y_train[CV_folds == j]

        knn = KNeighborsClassifier(n_neighbors=K)
        knn.fit(cv_X_train, cv_y_train)
        CV_accuracy[j-1] = knn.score(cv_X_val, cv_y_val)

    K_accuracy.append((K, np.mean(CV_accuracy))) # value of K and accuracy pairs

# value of K with the highest accuracy
best_K, best_acc = max(K_accuracy, key=lambda x: x[1])
print(f"\nResults from Cross Validation")
print(f"Best K: {best_K} and accuracy: {best_acc}")

# all the (K, accuracy) pairs for analysis purposes
print("\nAll accuracy values (K, accuracy):")
print(K_accuracy)

print("\nTraining Final Model for KNN...")
# FINAL KNN model using best K
knn = KNeighborsClassifier(n_neighbors=best_K)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_valid)
final_accuracy = knn.score(X_valid, y_valid)
final_precision = precision_score(y_valid, y_pred)
final_recall = recall_score(y_valid, y_pred)

print(f"\nFinal KNN model")
print(f"Accuracy: {final_accuracy}")
print(f"Precision: {final_precision}")
print(f"Recall: {final_recall}")