# Phase 4 – Implementation and Experimentation

## Model Training

Seven machine learning algorithms were trained and compared for the financial risk classification task: Logistic Regression (baseline), Decision Tree, Random Forest, Support Vector Machine (SVM), Naive Bayes, K-Nearest Neighbors (KNN), and Gradient Boosting. All models were implemented using scikit-learn and trained on the preprocessed dataset from Phase 2 (4,094 training samples, 53 features).

Hyperparameter tuning was performed using GridSearchCV with RepeatedStratifiedKFold cross-validation (5 folds, 3 repeats). For each model, a simple grid of 2-3 key hyperparameters was searched: Logistic Regression (C: [0.1, 1, 10], penalty: ['l2'], solver: ['lbfgs', 'sag']), Decision Tree (max_depth: [3, 5, 10, None], min_samples_split: [2, 5, 10]), Random Forest (n_estimators: [50, 100, 200], max_depth: [5, 10, None]), SVM (C: [0.1, 1, 10], kernel: ['rbf', 'linear']), Naive Bayes (var_smoothing: [1e-9, 1e-8, 1e-7]), KNN (n_neighbors: [3, 5, 7, 9], weights: ['uniform', 'distance']), and Gradient Boosting (n_estimators: [50, 100], learning_rate: [0.01, 0.1], max_depth: [3, 5]). The best hyperparameters were selected based on cross-validation accuracy scores.

To prevent overfitting, several techniques were employed: L2 regularization for Logistic Regression and SVM, tree depth limits for tree-based models, cross-validation for hyperparameter selection, ensemble methods (Random Forest and Gradient Boosting), and minimum samples for splitting in Decision Trees. All models were trained to completion without early stopping, and final evaluation was performed on the held-out test set (1,755 samples, 30% of the data).

## Model Evaluation

All models were evaluated on the test set using comprehensive metrics: accuracy, precision (per class and macro-averaged), recall (per class and macro-averaged), F1-score (per class and macro-averaged), and confusion matrices. Macro-averaged metrics were used to give equal weight to each class, which is appropriate given the class imbalance (Low: 55.8%, Medium: 23.2%, High: 21.0%).

The evaluation results show the following performance ranking on the test set:

1. **Gradient Boosting** - 99.60% accuracy (Precision: 99.56%, Recall: 99.57%, F1: 99.57%)
2. **Random Forest** - 99.37% accuracy (Precision: 99.34%, Recall: 99.39%, F1: 99.36%)
3. **Decision Tree** - 98.80% accuracy (Precision: 98.89%, Recall: 98.62%, F1: 98.75%)
4. **Support Vector Machine (SVM)** - 89.63% accuracy (Precision: 88.83%, Recall: 88.24%, F1: 88.49%)
5. **Logistic Regression** - 89.06% accuracy (Precision: 88.95%, Recall: 87.76%, F1: 88.33%)
6. **Naive Bayes** - 82.17% accuracy (Precision: 80.36%, Recall: 82.60%, F1: 81.14%)
7. **K-Nearest Neighbors (KNN)** - 78.97% accuracy (Precision: 77.85%, Recall: 71.32%, F1: 73.68%)

## Conclusions

Gradient Boosting achieved the highest accuracy (99.60%) and balanced performance across all metrics, demonstrating excellent performance for this financial risk classification problem. Random Forest (selected in Phase 3) performed very close to Gradient Boosting (99.37% vs 99.60%), confirming it as an excellent choice with better interpretability through feature importance measures, which are valuable in financial risk assessment contexts.

All tree-based ensemble methods (Gradient Boosting, Random Forest, Decision Tree) significantly outperformed linear models (SVM, Logistic Regression), indicating strong non-linear relationships between financial ratios and risk levels. All models achieved reasonable performance (>78% accuracy), validating the quality of the financial features selected in Phase 2. The confusion matrices revealed that most classification errors occurred between Medium and High risk categories, which is expected given their similar financial characteristics.

Cross-validation using RepeatedStratifiedKFold (5 folds, 3 repeats) was used extensively during hyperparameter tuning, providing 15 different train/validation splits for robust performance estimation. The stratified approach ensured that each fold maintained the same class distribution as the original dataset, which is crucial given the class imbalance.

