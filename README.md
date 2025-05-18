# MAGIC-Gamma-Telescope
This project compares ML classifiers—KNN, SVM &amp; Naive Bayes—on the MAGIC Gamma Telescope dataset (19,020 observations, 10 features). The goal: classify gamma rays vs hadronic showers, a complex task due to overlapping distributions &amp; non-linear patterns
# 🌌 Gamma Ray Classification with ML Classifiers

This project evaluates and compares the performance of **K-Nearest Neighbors (KNN)**, **Support Vector Machine (SVM)**, and **Naive Bayes** classifiers on the MAGIC Gamma Telescope dataset. The goal is to accurately classify cosmic events into **gamma rays (signal)** and **hadronic showers (background)**.

---

## Dataset Overview

- **Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/magic+gamma+telescope)
- **Size**: 19,020 observations
- **Features**: 10 continuous numerical variables representing physical measurements of detected particles
- **Task**: Binary classification  
  - **Class 0** – Hadronic showers (background)  
  - **Class 1** – Gamma rays (signal)

---

## Models & Key Findings

### 1️ K-Nearest Neighbors (KNN)
- **Feature Scaling**: StandardScaler
- **Hyperparameter Tuning**: GridSearchCV (5-fold)
- **Best Config**: `n_neighbors=11`, `weights=distance`
- **Performance**:  
  - Accuracy: **84.37%**
  - AUC: 0.90
  - Fast training & testing time  
-  Good balance of performance and efficiency

### 2 Naive Bayes
- **PCA Applied**: 95% variance retained (7 components)
- **Performance**:
  - Accuracy: **75.88%**
  - Fastest model (<0.005s runtime)
-  Struggles with the minority class (Class 1), despite PCA improvement

### 3️ Support Vector Machine (SVM)
- **Best Config**: `C=10`, `gamma=0.1`
- **With/without PCA tested**: PCA slightly reduced performance
- **Performance**:
  - Accuracy: **87.11%**
  - AUC: **0.93**
-  **Best overall performance**, though slowest to train

---

##  Summary Table

| Model       | Accuracy | AUC   | Training Time | Minority Class Recall | Best Use Case                    |
|-------------|----------|-------|----------------|------------------------|----------------------------------|
| SVM         | 87.11%   | 0.93  | ~2747 s        | ✅ Best                | Best performance, higher cost   |
| KNN         | 84.37%   | 0.90  | ~20 s          | Moderate               | Great trade-off of speed/accuracy |
| Naive Bayes | 75.88%   | ~0.85 | <0.005 s       | ❌ Weak                | When speed is critical           |

---

##  Conclusion

SVM emerged as the most effective model for this binary classification task due to its high accuracy and AUC, especially in distinguishing the minority class. KNN is a great fallback when speed is a concern. Naive Bayes, although efficient, underperforms in more complex decision boundaries.


 Author
Razieh Moradi Graduate Student, McMaster University  moradr1@mcmaster.ca



