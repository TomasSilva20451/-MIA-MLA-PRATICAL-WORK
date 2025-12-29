# Intelligent Financial Risk Classification for Small Businesses Using Machine Learning

## Project Overview

This project develops and evaluates a machine learning classification model to identify financial risk levels of small businesses using historical financial and economic indicators. The goal is to support early and informed decision-making for financial risk assessment.

**Problem Type:** Supervised Classification  
**Target:** 3-Class Financial Risk Classification (Low, Medium, High)  
**Framework:** MLA – MAAI Practical Work

## Objective

Develop and evaluate a machine learning classification model that identifies financial risk levels of small businesses using historical financial and economic indicators to support early and informed decision making.

## Phase 1. Problem Definition

### (a) What specific problem are you solving?

The problem addressed in this project is the automatic classification of small businesses into financial risk categories based on historical financial indicators. The goal is to identify companies that may be facing financial difficulties and classify them into Low, Medium, or High risk categories, supporting early and informed decision making.

This classification aims to help stakeholders such as financial institutions, investors, and business managers detect potential financial distress at an early stage and take appropriate preventive actions.

### (b) What type of ML task is it?

This problem is formulated as a **supervised machine learning classification task**.

More specifically, it is a **multi-class classification problem**, since the target variable consists of three discrete categories representing different levels of financial risk (Low, Medium, High). The use of supervised learning is justified because the dataset contains labeled examples where the financial risk outcome is known.

### (c) What is the expected outcome of the model?

The expected outcome of the model is to predict the financial risk level of a small business based on its financial ratios.

The model outputs:
- A predicted risk class (Low, Medium, or High) for each company
- Probability estimates for each risk category
- Information about the most relevant financial indicators contributing to the prediction

These outputs allow both risk classification and basic interpretability, which is important in a financial context.

## Phase 2. Data Collection & Preprocessing

### (a) What are the data sources?

The primary dataset used in this project is the **Polish Companies Bankruptcy Data**, obtained from the UCI Machine Learning Repository.

This dataset contains company-level financial information collected from small and medium-sized enterprises between 2000 and 2004. It includes 64 pre-computed financial ratios derived from balance sheets and income statements, covering liquidity, profitability, solvency, and efficiency aspects.

The original dataset provides a binary bankruptcy label, which was adapted in this project to a three-class financial risk system to better reflect different levels of financial risk.

**Dataset URL:** https://archive.ics.uci.edu/ml/datasets/Polish+Companies+Bankruptcy+Data

### (b) Are there missing values, duplicates, or outliers in the data?

Yes, data quality issues were identified and addressed:

- **Missing values:** Several financial ratios contained missing values. These were handled using median imputation, which is appropriate for numerical financial data and robust to extreme values.
- **Duplicates:** A small number of duplicate records (60 found) were found and removed to ensure that each company was represented only once.
- **Outliers:** Extreme values were observed in some financial ratios, which is common in financial datasets. To reduce their influence without removing information, winsorization was applied by clipping values at the 1st and 99th percentiles.

After these steps, the dataset was fully cleaned and suitable for modeling.

### (c) How were features selected, created, or transformed?

All original financial ratios were initially retained. To reduce redundancy, highly correlated features were identified, and those with very high correlation (> 0.95) were removed to avoid multicollinearity.

No new features were created, since the dataset already contains meaningful financial ratios.

For transformation:
- **Numerical features** were standardized using Z-score normalization so that all features have comparable scales.
- This step is important for algorithms that are sensitive to feature magnitude.

The final feature set consists of 53 financial ratios representing different aspects of financial health.

### (d) How were the data divided into training, validation, and test sets?

The dataset was divided into:
- **70% training data** (4,094 samples)
- **30% test data** (1,755 samples)

A stratified split was used to preserve the proportion of each risk class in both sets, which is important due to class imbalance.

No separate validation set was created. Instead, cross-validation was applied only on the training data during model selection. A fixed random seed (42) was used to ensure reproducibility.

All preprocessing steps such as imputation and scaling were fitted only on the training data and then applied to the test data, preventing data leakage.

## Dataset

**Source:** Polish Companies Bankruptcy Data from UCI Machine Learning Repository  
**URL:** https://archive.ics.uci.edu/ml/datasets/Polish+Companies+Bankruptcy+Data

- **Size:** 5,910 companies with 64 financial ratios
- **Time Period:** 2000-2004 (5 years)
- **Features:** Financial ratios covering liquidity, solvency, profitability, and efficiency metrics
- **Target:** Binary bankruptcy classification (adapted to 3-class risk levels)

## Project Status

- ✅ **Phase 1:** Project Definition and Planning
- ✅ **Phase 2:** Understanding & Preparation of Data 
- ✅ **Phase 3:** Model Selection and Training
- ✅ **Phase 4:** Implementation and Experimentation (Current)
- ⏳ **Phase 5:** Model Deployment and Monitoring (Upcoming)

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

```bash
# Clone the repository
git clone https://github.com/TomasSilva20451/-MIA-MLA-PRATICAL-WORK.git
cd "Phase 2 Understanding & Preparation of Data"

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Run Phase 2 Data Preparation

**Jupyter Notebook (Interactive)**
```bash
jupyter notebook Phase2_Data_Preparation.ipynb
```

The notebook will guide you through:
1. Load the dataset
2. Create 3-class risk categories (Low, Medium, High)
3. Clean the data (handle missing values, remove duplicates, treat outliers)
4. Prepare features (remove redundancy, encode, scale)
5. Split into train/test sets (70/30, stratified)
6. Save all outputs to appropriate directories

### Run Phase 4 Model Training

**Jupyter Notebook (Interactive)**
```bash
jupyter notebook Phase4_Implementation_Experimentation/Phase4_Model_Training.ipynb
```

The notebook will guide you through:
1. Load preprocessed training and test data
2. Define models and hyperparameter grids
3. Train all models with GridSearchCV
4. Evaluate models on test set
5. Compare models and select best model
6. Save results and visualizations

## Project Structure

```
.
├── src/                           # Source code
│   ├── config.py                  # Configuration constants
│   └── data/                      # Data processing modules
│       ├── load_data.py           # Data loading and risk categorization
│       ├── clean_data.py          # Missing values, duplicates, outliers
│       └── prepare_features.py    # Feature selection and scaling
├── data/
│   ├── raw/                       # Raw dataset files
│   ├── processed/                 # Cleaned dataset
│   └── splits/                    # Train/test splits
├── artifacts/                     # Preprocessing artifacts and model outputs
│   ├── models/                    # Trained models
│   ├── results/                   # Evaluation results
│   └── visualizations/            # Comparison visualizations
├── reports/                       # Academic reports
├── Phase2_Data_Preparation.ipynb  # Phase 2 data preparation notebook
├── Phase4_Implementation_Experimentation/
│   └── Phase4_Model_Training.ipynb # Phase 4 model training notebook
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## Phase 2 Results

### Data Processing Summary

- **Original Dataset:** 5,910 companies, 64 features
- **After Cleaning:** 5,849 samples (60 duplicates removed)
- **Final Features:** 53 features (12 redundant features removed)
- **Risk Distribution:**
  - Low Risk: 55.8% (3,300 companies)
  - Medium Risk: 23.2% (1,375 companies)
  - High Risk: 21.0% (1,235 companies)

### Data Splits

- **Training Set:** 4,094 samples (70%)
- **Test Set:** 1,755 samples (30%)
- **Stratification:** Applied to maintain class proportions

### Outputs Generated

- `data/processed/cleaned_data.csv` - Complete cleaned dataset
- `data/splits/X_train.csv, X_test.csv` - Feature matrices
- `data/splits/y_train.csv, y_test.csv` - Target variables
- `artifacts/scaler.joblib` - Fitted StandardScaler
- `artifacts/feature_list.joblib` - Selected feature list

## Methodology

### Data Cleaning
- **Missing Values:** Median imputation (numerical), mode imputation (categorical)
- **Duplicates:** Removed 60 duplicate rows
- **Outliers:** Winsorization (clipped at 1st and 99th percentiles)

### Feature Engineering
- **Redundancy Removal:** Features with correlation > 0.95 removed
- **Encoding:** One-hot encoding for categorical variables
- **Scaling:** StandardScaler (Z-score normalization)

### Data Splitting
- **Ratio:** 70% training / 30% test
- **Method:** Stratified splitting (maintains class proportions)
- **Reproducibility:** Fixed random seed (42)

## Phase 3: Model Selection & Justification

### (a) Baseline Model

The baseline model used in this work is **Logistic Regression**.

This model was selected because it is simple, easy to interpret, and commonly used as a reference model in classification problems, especially in financial risk assessment. Logistic Regression provides probability outputs and allows an initial understanding of how well the problem can be solved using a linear and interpretable approach.

### (b) Models Considered

In addition to the baseline model, the following machine learning models were considered and evaluated:

- **Decision Tree**, due to its interpretability and ability to model non-linear relationships
- **Random Forest**, as an ensemble method that reduces overfitting and improves predictive performance
- **Support Vector Machine (SVM)**, capable of handling complex decision boundaries
- **Naive Bayes**, a simple probabilistic classifier used for comparison
- **K Nearest Neighbors (KNN)**, an instance-based method suitable for standardized numerical features
- **Gradient Boosting**, an ensemble technique known for strong performance on structured data

Neural Networks were not considered at this stage because they require more data and tuning and offer lower interpretability, which is less suitable for an academic financial risk context.

### (c) Model Choice Justification

After comparing the models using cross-validation on the training data, **Random Forest** was selected as the final model.

This choice was justified by its strong performance on tabular financial data, robustness to noise and outliers, and its ability to capture non-linear relationships between financial indicators. Additionally, Random Forest provides feature importance measures, which improve model interpretability and support decision making in a financial risk assessment context.

Overall, Random Forest offered the best balance between predictive performance, robustness, and interpretability, making it the most appropriate model for this problem.

## Phase 4: Training the Model

### (a) What algorithm was used for training?

Seven machine learning algorithms were trained and compared:

1. **Logistic Regression** - Used as the baseline model due to its simplicity and interpretability
2. **Decision Tree** - A tree-based model that captures non-linear relationships
3. **Random Forest** - An ensemble method combining multiple decision trees
4. **Support Vector Machine (SVM)** - A kernel-based method for complex decision boundaries
5. **Naive Bayes** - A probabilistic classifier based on Bayes' theorem
6. **K-Nearest Neighbors (KNN)** - An instance-based learning method
7. **Gradient Boosting** - An ensemble technique that builds models sequentially

All models were implemented using scikit-learn and trained on the same preprocessed dataset.

### (b) What hyper-parameters were tuned and how?

Hyperparameter tuning was performed using **GridSearchCV** with **RepeatedStratifiedKFold** cross-validation (5 folds, 3 repeats). For each model, a simple grid of 2-3 key hyperparameters was searched:

- **Logistic Regression**: C values [0.1, 1, 10], penalty ['l2'], solver ['lbfgs', 'sag']
- **Decision Tree**: max_depth [3, 5, 10, None], min_samples_split [2, 5, 10]
- **Random Forest**: n_estimators [50, 100, 200], max_depth [5, 10, None]
- **SVM**: C [0.1, 1, 10], kernel ['rbf', 'linear']
- **Naive Bayes**: var_smoothing [1e-9, 1e-8, 1e-7]
- **KNN**: n_neighbors [3, 5, 7, 9], weights ['uniform', 'distance']
- **Gradient Boosting**: n_estimators [50, 100], learning_rate [0.01, 0.1], max_depth [3, 5]

The best hyperparameters were selected based on cross-validation accuracy scores. This approach ensures that hyperparameter selection is based on model performance across multiple validation folds, reducing overfitting.

### (c) What training strategy was implemented?

The training strategy followed a simple and standard approach:

1. **Cross-validation on training set**: RepeatedStratifiedKFold (5 folds, 3 repeats) was used to evaluate model performance during hyperparameter tuning. This ensures that class proportions are maintained in each fold, which is important given the class imbalance in the dataset.

2. **No separate validation set**: Instead of creating a separate validation set, cross-validation was applied directly on the training data. This maximizes the use of available training data while still providing robust performance estimates.

3. **No early stopping**: Early stopping was not used, as the models are relatively simple and training time is manageable. All models were trained to completion.

4. **Final evaluation on test set**: After hyperparameter tuning, the best model for each algorithm was evaluated on the held-out test set (30% of the data) to obtain final performance estimates.

This strategy is appropriate for an academic project and ensures reproducibility while maintaining a clear separation between training and testing data.

### (d) Were techniques used to handle overfitting?

Yes, several techniques were used to prevent overfitting:

1. **Regularization**: Logistic Regression uses L2 regularization (controlled by the C parameter) with multiclass-compatible solvers (lbfgs, sag). SVM uses L2 regularization to penalize large coefficients and reduce model complexity.

2. **Tree depth limits**: Decision Tree, Random Forest, and Gradient Boosting models use max_depth parameters to limit tree growth and prevent overfitting to training data.

3. **Cross-validation for hyperparameter selection**: By using cross-validation to select hyperparameters, the models are less likely to overfit to the training data, as performance is evaluated across multiple folds.

4. **Ensemble methods**: Random Forest and Gradient Boosting are ensemble methods that naturally reduce overfitting by combining multiple models.

5. **Minimum samples for splitting**: Decision Tree uses min_samples_split to prevent splitting on nodes with too few samples, reducing overfitting.

These techniques ensure that the models generalize well to unseen data while maintaining good performance on the training set.

## Phase 5: Model Evaluation

### (a) What metrics were used to evaluate the model?

Comprehensive evaluation metrics were computed for all models:

1. **Accuracy** - Overall proportion of correct predictions
2. **Precision** - Per class and macro-averaged (proportion of positive predictions that are correct)
3. **Recall** - Per class and macro-averaged (proportion of actual positives that are correctly identified)
4. **F1-Score** - Per class and macro-averaged (harmonic mean of precision and recall)
5. **Confusion Matrix** - Detailed breakdown of predictions vs. actual labels for each class

These metrics provide a complete picture of model performance, especially important for multi-class classification with class imbalance. Macro-averaged metrics give equal weight to each class, which is appropriate when all classes are equally important.

### (b) How did the model perform on the validation/test set?

All models were evaluated on the test set (1,755 samples, 30% of the data). The evaluation results show the following performance ranking:

1. **Gradient Boosting** - 99.60% accuracy (Precision: 99.56%, Recall: 99.57%, F1: 99.57%)
2. **Random Forest** - 99.37% accuracy (Precision: 99.34%, Recall: 99.39%, F1: 99.36%)
3. **Decision Tree** - 98.80% accuracy (Precision: 98.89%, Recall: 98.62%, F1: 98.75%)
4. **Support Vector Machine (SVM)** - 89.63% accuracy (Precision: 88.83%, Recall: 88.24%, F1: 88.49%)
5. **Logistic Regression** - 89.06% accuracy (Precision: 88.95%, Recall: 87.76%, F1: 88.33%)
6. **Naive Bayes** - 82.17% accuracy (Precision: 80.36%, Recall: 82.60%, F1: 81.14%)
7. **K-Nearest Neighbors (KNN)** - 78.97% accuracy (Precision: 77.85%, Recall: 71.32%, F1: 73.68%)

**Key Findings:**
- **Gradient Boosting** achieved the highest accuracy, demonstrating excellent performance for this financial risk classification problem
- **Random Forest** (selected in Phase 3) performed very close to Gradient Boosting (99.37% vs 99.60%), confirming it as an excellent choice with better interpretability
- All tree-based ensemble methods (Gradient Boosting, Random Forest, Decision Tree) significantly outperformed linear models, indicating strong non-linear relationships in the data
- All models achieved reasonable performance (>78% accuracy), validating the quality of the financial features
- The confusion matrices revealed which classes were most frequently confused, with most errors occurring between Medium and High risk categories

Detailed results are saved in `artifacts/results/model_comparison.csv` and `artifacts/results/model_evaluation_results.json`. Visualizations comparing all models are available in `artifacts/visualizations/`.

### (c) Was cross-validation used?

Yes, **RepeatedStratifiedKFold** cross-validation was used extensively:

1. **During hyperparameter tuning**: GridSearchCV used RepeatedStratifiedKFold (5 folds, 3 repeats) to evaluate each hyperparameter combination. This provides 15 different train/validation splits for robust performance estimation.

2. **For model selection**: Cross-validation scores were used to compare models and select the best hyperparameters for each algorithm.

3. **Stratification**: The stratified approach ensures that each fold maintains the same class distribution as the original dataset, which is crucial given the class imbalance (Low: 55.8%, Medium: 23.2%, High: 21.0%).

Cross-validation provides a more reliable estimate of model performance than a single train/validation split, especially important when working with limited data.

### (d) Were different models compared?

Yes, all seven models were trained, evaluated, and compared:

1. **Baseline comparison**: Logistic Regression served as a baseline to establish a reference performance level.

2. **Comprehensive comparison**: All models were evaluated on the same test set using the same metrics, allowing for direct comparison.

3. **Visual comparison**: Multiple visualizations were created:
   - Accuracy comparison bar chart
   - Precision/Recall/F1-score comparison
   - Confusion matrices for all models

4. **Best model selection**: Based on the comprehensive evaluation, **Gradient Boosting** achieved the highest accuracy (99.60%) and balanced performance across all metrics. However, **Random Forest** (selected in Phase 3) performed very close (99.37%) and remains an excellent choice due to its better interpretability, robustness, and feature importance measures, which are valuable in financial risk assessment contexts.

The comparison results are documented in the comparison table and saved to `artifacts/results/model_comparison.csv`. This systematic comparison ensures that the model selection is well-justified and reproducible.

## Configuration

Key settings can be modified in `src/config.py`:
- Train/test split ratio (default: 70/30)
- Random seed (default: 42)
- Correlation threshold for redundant features (default: 0.95)

## Documentation

- **Academic Report:** `reports/academic_report.md` - Phase 2 methodology summary
- **Phase 2 Notebook:** `Phase2_Data_Preparation.ipynb` - Interactive data preparation with visualizations
- **Phase 4 Notebook:** `Phase4_Implementation_Experimentation/Phase4_Model_Training.ipynb` - Model training and evaluation
- **Model Justification:** Phase 3 model selection and justification (see Phase 3 section above)

## Requirements

See `requirements.txt` for complete list. Main dependencies:
- numpy>=1.21.0
- pandas>=1.3.0
- scikit-learn>=1.0.0
- scipy>=1.7.0
- matplotlib>=3.5.0
- seaborn>=0.12.0
- jupyter>=1.0.0

## Next Steps

Phase 2, 3, and 4 are complete. The data is prepared, models are trained and evaluated. Ready for:
- **Phase 5:** Model Deployment and Monitoring (if applicable)
- Final report and presentation preparation

## Repository

**GitHub:** https://github.com/TomasSilva20451/-MIA-MLA-PRATICAL-WORK

## Author

**Tomás Silva**  
Academic Project - MLA/MAAI  
2025

## License

Academic project for educational purposes.
