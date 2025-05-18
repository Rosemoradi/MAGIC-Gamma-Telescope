import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest, mutual_info_classif

# Step 1: Load the dataset
file_path = r"C:\Users\Rose\Documents\mcmaster\semester 1\ML\project\magic04.xlsx"
data = pd.read_excel(file_path)

# Step 2: Prepare features (X) and labels (y)
X = data.iloc[:, :-1]  # All columns except the last one (features)
y = data.iloc[:, -1]   # Last column (target/labels)

# Step 3: Encode the labels (class: g -> 0, h -> 1)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)  # g -> 0, h -> 1

# Step 4: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 5: Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit and transform on training set
X_test_scaled = scaler.transform(X_test)        # Transform test set using the same scaler

# Step 6: Initialize variables for feature selection and Naive Bayes
accuracy_scores = []
k_values = range(1, X_train.shape[1] + 1)  # Iterate from 1 to the number of features
best_selector = None  # To store the selector with the best features

# Step 7: Perform feature selection and evaluate Naive Bayes
for k in k_values:
    # Select top k features based on Mutual Information
    selector = SelectKBest(score_func=mutual_info_classif, k=k)
    X_train_selected = selector.fit_transform(X_train_scaled, y_train)
    X_test_selected = selector.transform(X_test_scaled)

    # Initialize and train the Naive Bayes classifier
    nb = GaussianNB()
    nb.fit(X_train_selected, y_train)

    # Make predictions and evaluate the model
    y_pred = nb.predict(X_test_selected)
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_scores.append(accuracy)

    # Keep track of the selector with the best accuracy
    if accuracy == max(accuracy_scores):
        best_selector = selector

# Step 8: Find the optimal k with the highest accuracy
optimal_k = k_values[np.argmax(accuracy_scores)]
print(f"Optimal number of features: {optimal_k}")
print(f"Highest accuracy: {max(accuracy_scores):.2f}")

# Step 9: Get the names of the top features
selected_feature_indices = best_selector.get_support(indices=True)  # Indices of selected features
selected_feature_names = X.columns[selected_feature_indices]  # Map indices to feature names
print(f"Names of the top {optimal_k} features: {list(selected_feature_names)}")
