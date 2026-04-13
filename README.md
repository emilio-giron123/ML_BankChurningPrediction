# ML Bank Churning Prediction

This project predicts whether a bank customer will churn using historical customer data. The repository is organized so the churn workflow can be run end to end from one entry point, while still allowing each model to be run separately.

The project now uses the processed dataset in [`data/processed/customer_churn_processed.csv`](/C:/Users/Emilio/PycharmProjects/ML_BankChurningPrediction/data/processed/customer_churn_processed.csv) for downstream analysis and modeling. That file removes columns that should not be used for training:

- `RowNumber`
- `CustomerId`
- `Surname`
- `Complain`

## Project goal

The codebase is built to answer three main questions:

- Which customer attributes are most strongly related to churn
- How accurately can customer churn be predicted with machine learning
- Which model performs better for this classification problem

The two models currently used are:

- Logistic Regression
- Random Forest Classifier

This is a classification project, so the regression model used here is `logistic regression`, not linear regression.

## Current workflow

The full project flow is:

1. Load the processed churn dataset.
2. Print and save a text-based EDA report.
3. Generate the EDA visuals in `figures/output/`.
4. Run the feature-engineering / preprocessing workflow.
5. Train and compare logistic regression and random forest.
6. Run hold-out evaluation, 5-fold cross-validation, and threshold analysis.
7. Save model comparison text outputs and result visuals in `results/`.

## Project structure

```text
ML_BankChurningPrediction/
|- data/
|  |- raw/
|  |  |- Customer-Churn-Records.csv
|  |- processed/
|  |  |- customer_churn_processed.csv
|- figures/
|  |- output/
|- notebooks/
|  |- eda_analysis.ipynb
|  |- eda_visuals.ipynb
|  |- feature_engineering.ipynb
|  |- logistic_regression.ipynb
|  |- random_forest.ipynb
|- results/
|- src/
|  |- __init__.py
|  |- config.py
|  |- data_loader.py
|  |- eda_analysis.py
|  |- eda_visuals.py
|  |- evaluate.py
|  |- model_visuals.py
|  |- models.py
|  |- preprocessing.py
|- main.py
|- run_logistic_regression.py
|- run_random_forest.py
|- README.md
```

## What the main code does

### `main.py`

[`main.py`](/C:/Users/Emilio/PycharmProjects/ML_BankChurningPrediction/main.py) is the full-project entry point.

When you run it, it:

1. Runs the EDA text report
2. Generates the EDA visuals
3. Runs the feature-engineering workflow
4. Runs the full model comparison workflow
5. Prints the saved artifact paths

Use this file when you want the full project output in one run.

### `src/config.py`

[`src/config.py`](/C:/Users/Emilio/PycharmProjects/ML_BankChurningPrediction/src/config.py) stores shared configuration such as:

- Raw and processed dataset paths
- Output folder paths
- The target column, `Exited`
- Identifier columns
- Excluded predictor columns

### `src/data_loader.py`

[`src/data_loader.py`](/C:/Users/Emilio/PycharmProjects/ML_BankChurningPrediction/src/data_loader.py) loads CSV rows into simple Python dictionaries. It is used by the lightweight EDA code that does not depend on pandas.

### `src/eda_analysis.py`

[`src/eda_analysis.py`](/C:/Users/Emilio/PycharmProjects/ML_BankChurningPrediction/src/eda_analysis.py) creates the text-based exploratory analysis report.

It prints and saves:

- First 5 rows
- Shape
- Column names
- Inferred data types
- Summary statistics
- Quartiles and IQR (min/ax, median, mean, std)
- Unique values per column
- Useless columns
- Strongest churn-related features

The output is saved to [`results/analysis_report.txt`](/C:/Users/Emilio/PycharmProjects/ML_BankChurningPrediction/results/analysis_report.txt).

### `src/eda_visuals.py`

[`src/eda_visuals.py`](/C:/Users/Emilio/PycharmProjects/ML_BankChurningPrediction/src/eda_visuals.py) creates the project EDA figures and saves them in [`figures/output/`](/C:/Users/Emilio/PycharmProjects/ML_BankChurningPrediction/figures/output).

Current EDA figures:

- `churn_distribution.svg`
- `geography_churned.svg`
- `age_vs_churn.svg`
- `is_active_member_vs_churn.svg`

### `src/preprocessing.py`

[`src/preprocessing.py`](/C:/Users/Emilio/PycharmProjects/ML_BankChurningPrediction/src/preprocessing.py) handles the feature-engineering and preprocessing setup used before modeling.

Its main responsibilities are:

- Remove excluded columns
- Split the dataset into features and target
- Define categorical and numeric columns
- One-hot encode categorical variables
- Scale numeric variables
- Save the processed CSV
- Create the train/test split used for modeling

The feature-engineering summary is saved to [`results/feature_engineering_summary.txt`](/C:/Users/Emilio/PycharmProjects/ML_BankChurningPrediction/results/feature_engineering_summary.txt).

### `src/evaluate.py`

[`src/evaluate.py`](/C:/Users/Emilio/PycharmProjects/ML_BankChurningPrediction/src/evaluate.py) contains the model-evaluation helpers.

It currently supports:

- Hold-out evaluation
- Confusion matrix generation
- ROC-AUC calculation
- ROC curve data generation
- 5-fold stratified cross-validation
- Threshold analysis

### `src/model_visuals.py`

[`src/model_visuals.py`](/C:/Users/Emilio/PycharmProjects/ML_BankChurningPrediction/src/model_visuals.py) creates the model-evaluation visuals saved inside [`results/`](/C:/Users/Emilio/PycharmProjects/ML_BankChurningPrediction/results).

It creates:

- Model performance comparison
- Confusion matrix analysis
- ROC curve analysis
- 5-fold cross-validation ROC-AUC graph
- Metrics vs threshold graph
- Random forest top 12 feature importance graph

### `src/models.py`

[`src/models.py`](/C:/Users/Emilio/PycharmProjects/ML_BankChurningPrediction/src/models.py) contains the machine-learning workflow.

It includes:

- logistic regression pipeline
- random forest pipeline
- shared train/test split helper
- standalone logistic regression workflow
- standalone random forest workflow
- full side-by-side comparison workflow

The comparison workflow handles:

- Hold-out test metrics
- Accuracy, precision, recall, F1-score, and ROC-AUC
- 5-fold cross-validation
- Threshold analysis
- Result visual generation

### Logistic Regression

[`run_logistic_regression.py`](/C:/Users/Emilio/PycharmProjects/ML_BankChurningPrediction/run_logistic_regression.py) runs only the logistic regression workflow.

It saves:

- [`results/logistic_regression_metrics.txt`](/C:/Users/Emilio/PycharmProjects/ML_BankChurningPrediction/results/logistic_regression_metrics.txt)
- hold-out metrics
- 5-fold cross-validation summary
- threshold analysis

### Random Forest 

[`run_random_forest.py`](/C:/Users/Emilio/PycharmProjects/ML_BankChurningPrediction/run_random_forest.py) runs only the random forest workflow.

It saves:

- [`results/random_forest_metrics.txt`](/C:/Users/Emilio/PycharmProjects/ML_BankChurningPrediction/results/random_forest_metrics.txt)
- [`results/random_forest_top_12_feature_importance.svg`](/C:/Users/Emilio/PycharmProjects/ML_BankChurningPrediction/results/random_forest_top_12_feature_importance.svg)
- hold-out metrics
- 5-fold cross-validation summary
- threshold analysis
- random forest feature-importance visualization

## Full comparison outputs

When you run [`main.py`](/C:/Users/Emilio/PycharmProjects/ML_BankChurningPrediction/main.py) or call the comparison workflow, the main model comparison text summary is saved to:

- [`results/model_comparison.txt`](/C:/Users/Emilio/PycharmProjects/ML_BankChurningPrediction/results/model_comparison.txt)

The comparison visuals saved in [`results/`](/C:/Users/Emilio/PycharmProjects/ML_BankChurningPrediction/results) are:

- [`results/model_performance_comparison.svg`](/C:/Users/Emilio/PycharmProjects/ML_BankChurningPrediction/results/model_performance_comparison.svg)
- [`results/confusion_matrix_analysis.svg`](/C:/Users/Emilio/PycharmProjects/ML_BankChurningPrediction/results/confusion_matrix_analysis.svg)
- [`results/roc_curve_analysis.svg`](/C:/Users/Emilio/PycharmProjects/ML_BankChurningPrediction/results/roc_curve_analysis.svg)
- [`results/five_fold_cv_roc_auc.svg`](/C:/Users/Emilio/PycharmProjects/ML_BankChurningPrediction/results/five_fold_cv_roc_auc.svg)
- [`results/vs_threshold_analysis.svg`](/C:/Users/Emilio/PycharmProjects/ML_BankChurningPrediction/results/vs_threshold_analysis.svg)
- [`results/random_forest_top_12_feature_importance.svg`](/C:/Users/Emilio/PycharmProjects/ML_BankChurningPrediction/results/random_forest_top_12_feature_importance.svg)

These visuals show:

- Side-by-side model metric performance
- Confusion matrices for both models
- ROC curves for both models
- ROC-AUC stability across folds
- How precision, recall, and F1 change across thresholds
- The most important random forest features

## Notebooks

The notebooks mirror the main stages of the project and make the workflow easier to follow interactively.

- [`notebooks/eda_analysis.ipynb`](/C:/Users/Emilio/PycharmProjects/ML_BankChurningPrediction/notebooks/eda_analysis.ipynb): text-based exploratory analysis
- [`notebooks/eda_visuals.ipynb`](/C:/Users/Emilio/PycharmProjects/ML_BankChurningPrediction/notebooks/eda_visuals.ipynb): EDA visual generation
- [`notebooks/feature_engineering.ipynb`](/C:/Users/Emilio/PycharmProjects/ML_BankChurningPrediction/notebooks/feature_engineering.ipynb): processed data and preprocessing steps
- [`notebooks/logistic_regression.ipynb`](/C:/Users/Emilio/PycharmProjects/ML_BankChurningPrediction/notebooks/logistic_regression.ipynb): logistic regression workflow
- [`notebooks/random_forest.ipynb`](/C:/Users/Emilio/PycharmProjects/ML_BankChurningPrediction/notebooks/random_forest.ipynb): random forest workflow

## How to run

### Run the full workflow

```powershell
python main.py
```

### Run only logistic regression

```powershell
python run_logistic_regression.py
```

### Run only random forest

```powershell
python run_random_forest.py
```

## Current outputs in `results/`

Text outputs:

- `analysis_report.txt`
- `feature_engineering_summary.txt`
- `logistic_regression_metrics.txt`
- `random_forest_metrics.txt`
- `model_metrics.txt`
- `model_comparison.txt`

Model visual outputs:

- `model_performance_comparison.svg`
- `confusion_matrix_analysis.svg`
- `roc_curve_analysis.svg`
- `five_fold_cv_roc_auc.svg`
- `vs_threshold_analysis.svg`
- `random_forest_top_12_feature_importance.svg`

## Current outputs in `figures/output/`

- `churn_distribution.svg`
- `geography_churned.svg`
- `age_vs_churn.svg`
- `is_active_member_vs_churn.svg`
