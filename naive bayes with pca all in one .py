import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import time
from sklearn.preprocessing import LabelBinarizer

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
X_train_scaled = scaler.fit_transform(X_train)  # Fit and transform the training set
X_test_scaled = scaler.transform(X_test)        # Transform the test set using the same scaler

# Step 6: Apply PCA for dimensionality reduction
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

# Step 7: Train the Gaussian Naïve Bayes model
nb_model = GaussianNB()

start_train = time.time()  # Start timing the training process
nb_model.fit(X_train_pca, y_train)
end_train = time.time()  # End timing the training process

# Step 8: Test the model and make predictions
start_test = time.time()  # Start timing the testing process
y_pred = nb_model.predict(X_test_pca)
end_test = time.time()  # End timing the testing process

# Step 9: Evaluate the model
print("\nTraining Time (seconds):", end_train - start_train)
print("Testing Time (seconds):", end_test - start_test)
print("\nTest Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Step 10: Plot the ROC Curve
lb = LabelBinarizer()
y_test_binary = lb.fit_transform(y_test)

# Predict probabilities for ROC curve
y_score = nb_model.predict_proba(X_test_pca)[:, 1]  # Use probabilities for class 1

fpr, tpr, _ = roc_curve(y_test_binary, y_score)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (Naïve Bayes with PCA)')
plt.legend(loc="lower right")
plt.show()
