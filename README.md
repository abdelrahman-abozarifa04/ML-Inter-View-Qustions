# ML Interview Questions

---

### 1. **What is Machine Learning?**

ML is a subset of AI where systems learn from data to improve performance on a task without being explicitly programmed.
📌 **Example**: Netflix recommending movies based on your watch history.
📦 **Library**: `scikit-learn`, `TensorFlow`, `PyTorch`.

---

### 2. **Supervised vs Unsupervised vs Reinforcement Learning**

| Type          | Description                            | Real-World Example    | Library                  |
| ------------- | -------------------------------------- | --------------------- | ------------------------ |
| Supervised    | Labeled data, maps input to output.    | Email spam detection  | `scikit-learn`           |
| Unsupervised  | Unlabeled data, finds hidden patterns. | Customer segmentation | `scikit-learn`, `KMeans` |
| Reinforcement | Learns via rewards/punishment.         | AlphaGo, Robotics     | `OpenAI Gym`             |

---

### 3. **What is Overfitting?**

A model performs well on training data but poorly on unseen data.

🧠 **Avoid it by**:

* Regularization (`L1`, `L2`)
* Cross-validation
* Pruning (for trees)
* Dropout (in neural networks)
* Early stopping

---

### 4. **Bias-Variance Tradeoff**

* **High bias**: Model is too simple → underfits.
* **High variance**: Model is too complex → overfits.

✅ Ideal Model: **Low bias + low variance** (balance).

---

### 5. **Parametric vs Non-Parametric**

| Type           | Meaning                    | Example              | Time Complexity      |
| -------------- | -------------------------- | -------------------- | -------------------- |
| Parametric     | Fixed number of parameters | Logistic Regression  | Fast                 |
| Non-Parametric | Grows with data            | K-NN, Decision Trees | Slower, but flexible |

---

### 6. **Training, Validation, and Test Sets**

* **Training**: Learn patterns
* **Validation**: Tune model (hyperparameter tuning)
* **Test**: Final evaluation

🧪 Split: Typical ratios are 60-20-20 or 70-15-15.

---

### 7. **Cross-Validation**

Method to evaluate models on different data folds to ensure generalization.
✅ Most common: **K-Fold CV**.
📦 Library: `cross_val_score` in `scikit-learn`.

---

### 8. **Precision, Recall, F1-Score**

* **Precision** = TP / (TP + FP)
* **Recall** = TP / (TP + FN)
* **F1** = 2 × (Precision × Recall) / (Precision + Recall)

📌 **Use F1-score** when you care about **both** false positives and false negatives.

---

### 9. **Confusion Matrix**

A 2×2 matrix for binary classification showing:

|            | Predicted Yes | Predicted No |
| ---------- | ------------- | ------------ |
| Actual Yes | TP            | FN           |
| Actual No  | FP            | TN           |

📦 Use: `sklearn.metrics.confusion_matrix()`

---

### 10. **Gradient Descent**

An optimization algorithm that adjusts parameters by following the **negative gradient** of the cost function.

🧮 Formula:

> θ = θ - α \* ∇J(θ)
> Where:

* θ = parameters
* α = learning rate

---

### 11. **Purpose of Cost Function**

Measures the error of the model.
📌 Goal: Minimize the cost function during training.

🧮 Example:

* MSE for regression
* Cross-entropy for classification

---

### 12. **Regularization**

Prevents overfitting by adding penalties to the cost function.

* **L1 (Lasso)**: Encourages sparsity
* **L2 (Ridge)**: Shrinks coefficients

📦 `sklearn.linear_model.Ridge` or `Lasso`

---

### 13. **Feature Engineering**

Transforming raw data into features to improve model performance.

Examples:

* One-hot encoding
* Polynomial features
* Log scaling
  📦 `sklearn.preprocessing`, `pandas`

---

### 14. **Curse of Dimensionality**

As features increase:

* Distance metrics become meaningless
* Data becomes sparse

📌 Solutions:

* Feature selection
* Dimensionality reduction (e.g., PCA)

---

### 15. **Bagging vs Boosting**

| Technique | Description                  | Example           | Runtime               |
| --------- | ---------------------------- | ----------------- | --------------------- |
| Bagging   | Combines models in parallel  | Random Forest     | Fast                  |
| Boosting  | Sequentially improves errors | XGBoost, AdaBoost | Slower, more accurate |

---

### 16. **PCA (Principal Component Analysis)**

Reduces dimensionality by projecting data into new axes (principal components) that capture max variance.

🧮 Steps:

1. Center data
2. Compute covariance matrix
3. Compute eigenvalues/eigenvectors
4. Project data

📦 `sklearn.decomposition.PCA`

---

### 17. **Ensemble Learning Methods**

Combine multiple models to improve accuracy.

Types:

* **Bagging** (Random Forest)
* **Boosting** (XGBoost)
* **Stacking** (meta-model on top of base models)

---

### 18. **Naive Bayes**

Based on **Bayes’ Theorem**, assumes feature independence.

🧮 P(y|x) ∝ P(x|y) × P(y)
📌 Fast and good for text classification.

📦 `sklearn.naive_bayes`

---

### 19. **K-Means Clustering**

Clusters data into **K** groups by minimizing within-cluster distance.

Steps:

1. Randomly choose K centroids
2. Assign points to nearest centroid
3. Recompute centroids

📦 `sklearn.cluster.KMeans`

---

### 20. **Random Forest vs Gradient Boosting**

| Model             | Strength             | Weakness             |
| ----------------- | -------------------- | -------------------- |
| Random Forest     | Robust, low variance | Less accurate        |
| Gradient Boosting | High accuracy        | Prone to overfitting |

---

### 21. **Support Vector Machines (SVM)**

Finds the best decision boundary (hyperplane) with maximum margin.

📦 `sklearn.svm.SVC`
👍 Good for high-dimensional data
👎 Slow on large datasets

---

### 22. **Logistic Regression for Multi-Class?**

Yes. Using:

* One-vs-Rest (OvR)
* Softmax (multinomial)

📦 `sklearn.linear_model.LogisticRegression(multi_class='multinomial')`

---

### 23. **Loss vs Cost Function**

* **Loss Function**: Error for one sample
* **Cost Function**: Average error over all samples

🧮 Example:

* Loss: `log loss for one point`
* Cost: `sum of all losses / N`

---

### 24. **Early Stopping**

Stops training when validation loss stops improving to prevent overfitting.

📦 Use in: `XGBoost`, `Keras`, `LightGBM`

---

### 25. **AUC-ROC**

* **ROC Curve**: TPR vs FPR at various thresholds
* **AUC**: Probability model ranks positive > negative

📌 AUC = 1 → perfect classifier
📦 `sklearn.metrics.roc_auc_score`

---

### 26. **How Random Forest Handles Overfitting**

* Uses bootstrapped datasets
* Random feature selection per tree
* Averages predictions → reduces variance

---

### 27. **Learning Rate in Gradient Descent**

Controls **step size**:

* Too small → slow
* Too large → overshoot

📌 Tune using grid search or learning rate schedules.

---

### 28. **SVM Intuition**

* Maximizes margin
* Can use **kernels** for non-linear classification (e.g., RBF, polynomial)

📦 `sklearn.svm.SVC(kernel='rbf')`

---

### 29. **KNN vs K-Means**

| Criteria | KNN                   | K-Means              |
| -------- | --------------------- | -------------------- |
| Type     | Supervised            | Unsupervised         |
| Use      | Classification        | Clustering           |
| Metric   | Distance to neighbors | Distance to centroid |

---

### 30. **Batch vs Stochastic Gradient Descent**

| Type  | Description          | Pros   | Cons  |
| ----- | -------------------- | ------ | ----- |
| Batch | Uses all data        | Stable | Slow  |
| SGD   | One sample at a time | Fast   | Noisy |

---

### 31. **Handling Imbalanced Datasets**

* **Resampling**: Oversample minority or undersample majority
* **SMOTE**: Create synthetic samples
* **Use metrics**: F1, ROC-AUC
* **Class weights**: `class_weight='balanced'`

---

### 32. **Handling Outliers**

**Detection:**

* Z-score, IQR
* Boxplot

**Treatment:**

* Remove if error
* Cap at percentiles (e.g., 1%, 99%)
* Transform: log, sqrt
* Use tree models which are robust

