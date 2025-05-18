import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import time
from sklearn.preprocessing import LabelBinarizer

# Step 1: Load the dataset
file_path = r"C:\Users\Rose\Documents\mcmaster\semester 1\ML\project\magic04.xlsx"
data = pd.read_excel(file_path)

# Step 2: Prepare features (X) and target (y)
X = data.iloc[:, :-1]  # All columns except the last one (features)
y = data.iloc[:, -1]   # Last column (labels)

# Step 3: Encode the labels (g -> 0, h -> 1)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Step 4: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# # Step 5: Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # Fit to training data and transform it
X_test = scaler.transform(X_test)        # Transform test data using training parameters

# Step 6: Set up the KNN model and parameter grid
knn_model = KNeighborsClassifier()
param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11],  # Number of neighbors to consider
    'weights': ['uniform', 'distance']  # Weighting function
}

# Step 7: Perform Grid Search with Cross-Validation
print("\nPerforming 5-Fold Cross-Validation for KNN Hyperparameter Selection...")
start_train = time.time()  # Start timing the training process
grid_search = GridSearchCV(estimator=knn_model, param_grid=param_grid, cv=5, scoring='accuracy', verbose=1)
grid_search.fit(X_train, y_train)
end_train = time.time()  # End timing the training process

# Step 8: Evaluate the best model on the test set
best_model = grid_search.best_estimator_  # Best parameters selected
start_test = time.time()  # Start timing the testing process
y_pred = best_model.predict(X_test)
end_test = time.time()  # End timing the testing process

# Step 9: Evaluate Performance Metrics
print("\nBest Hyperparameters:", grid_search.best_params_)
print("\nTraining Time (seconds):", end_train - start_train)
print("Testing Time (seconds):", end_test - start_test)
print("\nTest Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Step 10: Plot the ROC Curve
lb = LabelBinarizer()
y_test_binary = lb.fit_transform(y_test)

# Predict probabilities for ROC curve
y_score = best_model.predict_proba(X_test)[:, 1]  # Use probabilities for class 1

fpr, tpr, _ = roc_curve(y_test_binary, y_score)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (KNN)')
plt.legend(loc="lower right")
plt.show()
