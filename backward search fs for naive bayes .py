import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Step 1: Load the dataset
file_path = r"C:\Users\Rose\Documents\mcmaster\semester 1\ML\project\magic04.xlsx"
data = pd.read_excel(file_path)

# Step 2: Prepare features (X) and labels (y)
X = data.iloc[:, :-1]  # All columns except the last one (features)
y = data.iloc[:, -1]  # Last column (target/labels)

# Step 3: Encode the labels (class: g -> 0, h -> 1)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)  # g -> 0, h -> 1

# Step 4: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 5: Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit and transform on training set
X_test_scaled = scaler.transform(X_test)  # Transform test set using the same scaler

# Step 6: Initialize variables
remaining_features = list(X.columns)  # Start with all features
best_accuracy = 0
best_features = remaining_features.copy()

# Step 7: Perform backward feature elimination
while len(remaining_features) > 1:
    accuracies = []
    for feature in remaining_features:
        # Remove one feature
        features_to_keep = [f for f in remaining_features if f != feature]
        X_train_reduced = X_train_scaled[:, [X.columns.get_loc(f) for f in features_to_keep]]
        X_test_reduced = X_test_scaled[:, [X.columns.get_loc(f) for f in features_to_keep]]

        # Train the Naive Bayes classifier
        nb = GaussianNB()
        nb.fit(X_train_reduced, y_train)
        y_pred = nb.predict(X_test_reduced)

        # Evaluate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append((feature, accuracy))

    # Find the feature whose removal gives the highest accuracy
    feature_to_remove, max_accuracy = min(accuracies, key=lambda x: x[1])

    # Update remaining features and best accuracy if improvement is seen
    if max_accuracy > best_accuracy:
        best_accuracy = max_accuracy
        remaining_features.remove(feature_to_remove)
        best_features = remaining_features.copy()
    else:
        break  # Stop if removing features no longer improves performance

# Step 8: Output the results
print(f"Optimal subset of features: {best_features}")
print(f"Accuracy with optimal subset: {best_accuracy:.2f}")
