import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score

X = np.load('X_processed.npy')  
y = np.load('y_processed.npy')  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

print(f"Training: {X_train.shape[0]}, Test: {X_test.shape[0]}, Features: {X_train.shape[1]}")

tree = DecisionTreeClassifier(max_depth=7, random_state=42)

tree.fit(X_train, y_train)

predictions = tree.predict(X_test)
score = accuracy_score(y_test, predictions)

print(f"Accuracy: {score:.4f}")
print(f"Normal Precision: {precision_score(y_test, predictions, pos_label=0):.4f}")
print(f"Attack Precision: {precision_score(y_test, predictions, pos_label=1):.4f}")
print(f"Normal Recall: {recall_score(y_test, predictions, pos_label=0):.4f}")
print(f"Attack Recall: {recall_score(y_test, predictions, pos_label=1):.4f}")

feature_imp = tree.feature_importances_
best_features = np.argsort(feature_imp)[::-1][:5]

print("\nMost Important Features:")
for i in range(5):
    feature_num = best_features[i]
    importance_value = feature_imp[feature_num]
    print(f"{i+1}. Feature {feature_num}: {importance_value:.4f}")

print(f"Tree depth: {tree.get_depth()}, Leaves: {tree.get_n_leaves()}")

print("\nTrying without Feature 7:")

X_train_new = np.delete(X_train, 7, axis=1)
X_test_new = np.delete(X_test, 7, axis=1)

tree2 = DecisionTreeClassifier(max_depth=7, random_state=42)

tree2.fit(X_train_new, y_train)

predictions2 = tree2.predict(X_test_new)
score2 = accuracy_score(y_test, predictions2)

print(f"Accuracy: {score2:.4f}")
print(f"Normal Precision: {precision_score(y_test, predictions2, pos_label=0):.4f}")
print(f"Attack Precision: {precision_score(y_test, predictions2, pos_label=1):.4f}")
print(f"Normal Recall: {recall_score(y_test, predictions2, pos_label=0):.4f}")
print(f"Attack Recall: {recall_score(y_test, predictions2, pos_label=1):.4f}\n")

feature_imp2 = tree2.feature_importances_
best_features2 = np.argsort(feature_imp2)[::-1][:5]

print("Most Important Features (without Feature 7):")
for i in range(5):
    feature_position = best_features2[i]
    if feature_position >= 7:
        original_feature_num = feature_position + 1
    else:
        original_feature_num = feature_position
    importance_value = feature_imp2[feature_position]
    print(f"{i+1}. Feature {original_feature_num}: {importance_value:.4f}")

print(f"Tree depth: {tree2.get_depth()}, Leaves: {tree2.get_n_leaves()}")

print(f"\nComparison:")
print(f"With Feature 7: {score:.4f}")
print(f"Without Feature 7: {score2:.4f}")

