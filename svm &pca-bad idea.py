import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import time

# Step 1: Load the dataset
file_path = r"C:\Users\Rose\Documents\mcmaster\semester 1\ML\project\magic04.xlsx"
data = pd.read_excel(file_path)

# Step 2: Prepare features (X) and label (y)
X = data.iloc[:, :-1]  # All columns except the last one (features)
y = data.iloc[:, -1]   # Last column (labels)

# Step 3: Encode the labels (g -> 0, h -> 1)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Step 4: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)

# Step 5: Normalize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit to training data and transform it
X_test_scaled = scaler.transform(X_test)        # Transform test data using the same scaler

# Step 6: Apply PCA
pca = PCA()
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Analyze explained variance
explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)
print("Explained Variance Ratio:", explained_variance)
print("Cumulative Variance Ratio:", cumulative_variance)

# Select the number of components to retain 95% variance
n_components = np.argmax(cumulative_variance >= 0.95) + 1
print(f"Number of components to retain 95% variance: {n_components}")

# Refit PCA with the optimal number of components
pca = PCA(n_components=n_components)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Step 7: Set up the SVM model and parameter grid
svm_model = SVC(kernel='rbf', probability=True)  # RBF kernel for non-linear SVM
param_grid = {
    'C': [0.1, 1, 10, 100],  # Regularization parameter
    'gamma': [0.01, 0.1, 1, 'scale']  # Kernel coefficient
}

# Perform Grid Search with Cross-Validation
print("\nPerforming Grid Search with Cross-Validation...")
start_train = time.time()  # Start timing the training process
grid_search = GridSearchCV(estimator=svm_model, param_grid=param_grid, cv=5, scoring='accuracy', verbose=1)
grid_search.fit(X_train_pca, y_train)
end_train = time.time()  # End timing the training process

# Step 8: Evaluate the best model
best_model = grid_search.best_estimator_  # Best parameters found
start_test = time.time()  # Start timing the testing process
y_pred = best_model.predict(X_test_pca)  # Predict on test set
end_test = time.time()  # End timing the testing process

# Display results
print("\nBest Hyperparameters:", grid_search.best_params_)
print("\nTest Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

print("\nTraining Time (seconds):", end_train - start_train)
print("Testing Time (seconds):", end_test - start_test)

# Step 9: Compute ROC Curve and AUC
y_score = best_model.decision_function(X_test_pca)

# Compute ROC curve and AUC for each class
fpr, tpr, _ = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)

# Plot ROC Curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (SVM with PCA)')
plt.legend(loc="lower right")
plt.show()
