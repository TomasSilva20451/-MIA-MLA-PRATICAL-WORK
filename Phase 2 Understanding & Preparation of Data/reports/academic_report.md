# Phase 2 – Understanding & Preparation of Data

## Data Source

The dataset used is the **Polish Companies Bankruptcy Data** from the UCI Machine Learning Repository. This dataset contains financial ratios from 5,910 Polish companies collected over five years (2000-2004). The dataset includes 64 pre-computed financial ratios covering liquidity, solvency, profitability, and efficiency metrics. The original binary bankruptcy classification (bankrupt/non-bankrupt) is adapted to a three-class risk system: Low Risk (healthy companies, 55.8%), Medium Risk (moderate distress, 23.2%), and High Risk (bankrupt or severe distress, 21.0%).

## Cleaning Decisions

Missing values are handled using median imputation for numerical features and mode imputation for categorical features. This simple approach is appropriate for academic work and preserves data integrity. Duplicate rows (60 found) are removed to prevent overfitting. Outliers are handled by clipping extreme values at the 1st and 99th percentiles (winsorization), preserving information while reducing the influence of extreme values.

## Features Used

All 64 financial ratios are initially retained. Highly correlated features (correlation > 0.95) are removed to reduce redundancy, resulting in 53 features. Categorical variables, if any, are one-hot encoded. All numerical features are standardized using StandardScaler to ensure comparable scales across different financial ratios.

## Split Strategy

The data is split into training (70%) and test (30%) sets using stratified splitting to maintain proportional representation of risk levels across both sets. A fixed random seed (42) ensures reproducibility. The stratified approach is essential given the class distribution (Low: 55.8%, Medium: 23.2%, High: 21.0%) to ensure fair model evaluation. After duplicate removal and preprocessing, the final dataset contains 5,849 samples, split into 4,094 training samples and 1,755 test samples.

