# ML Bank Churning Prediction

This project predicts whether a bank customer will churn by using historical customer records. The workflow is organized so the data can be explored, cleaned, visualized, and modeled in separate stages, while still allowing the full pipeline to run from one entry point.

The current project uses the processed dataset in [`data/processed/customer_churn_processed.csv`](/C:/Users/Emilio/PycharmProjects/ML_BankChurningPrediction/data/processed/customer_churn_processed.csv) for downstream analysis and modeling. That processed file removes columns that should not be used for training:

- `RowNumber`
- `CustomerId`
- `Surname`
- `Complain`

## Current workflow

The project follows this order:

1. Load the processed churn dataset.
2. Run a text-based exploratory data analysis report.
3. Generate the EDA visuals in `figures/output/`.
4. Run the feature-engineering workflow to confirm the processed dataset and preprocessing pipeline.
5. Train and evaluate the two classification models:
   - Logistic Regression
   - Random Forest Classifier

All saved text outputs go into [`results/`](/C:/Users/Emilio/PycharmProjects/ML_BankChurningPrediction/results), and all chart outputs go into [`figures/output/`](/C:/Users/Emilio/PycharmProjects/ML_BankChurningPrediction/figures/output).

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
|  |- models.py
|  |- preprocessing.py
|- main.py
|- run_logistic_regression.py
|- run_random_forest.py
|- README.md
```

## What each main file does

### `main.py`

[`main.py`](/C:/Users/Emilio/PycharmProjects/ML_BankChurningPrediction/main.py) is the full-project entry point.

When you run it, it:

1. runs the EDA text report
2. generates the SVG visuals
3. runs the feature-engineering workflow
4. trains and evaluates both models
5. prints the saved output paths

Use this file when you want the complete workflow in one run.

### `src/config.py`

[`src/config.py`](/C:/Users/Emilio/PycharmProjects/ML_BankChurningPrediction/src/config.py) stores the shared configuration used across the project.

It defines:

- project root paths
- raw and processed data paths
- output folders for figures and results
- the churn target column, `Exited`
- identifier columns that should not be treated as useful predictors
- predictor columns that should be excluded from churn ranking

This file keeps paths and column rules centralized so the rest of the project does not hardcode them repeatedly.

### `src/data_loader.py`

[`src/data_loader.py`](/C:/Users/Emilio/PycharmProjects/ML_BankChurningPrediction/src/data_loader.py) reads the CSV file and converts it into row dictionaries. This is used by the lightweight EDA modules that do not need pandas.

### `src/eda_analysis.py`

[`src/eda_analysis.py`](/C:/Users/Emilio/PycharmProjects/ML_BankChurningPrediction/src/eda_analysis.py) builds the text-based exploration report.

It prints and saves:

- first 5 rows
- shape
- column names
- inferred data types
- summary statistics
- quartile information including `Q1`, `Q3`, and `IQR`
- unique values per column
- likely useless columns
- strongest churn predictors

The saved report is written to [`results/analysis_report.txt`](/C:/Users/Emilio/PycharmProjects/ML_BankChurningPrediction/results/analysis_report.txt).

### `src/eda_visuals.py`

[`src/eda_visuals.py`](/C:/Users/Emilio/PycharmProjects/ML_BankChurningPrediction/src/eda_visuals.py) creates the project's EDA charts and saves them as SVG files.

The current visuals include:

- `churn_distribution.svg`
- `geography_churned.svg`
- `age_vs_churn.svg`
- `is_active_member_vs_churn.svg`

These visuals are saved in [`figures/output/`](/C:/Users/Emilio/PycharmProjects/ML_BankChurningPrediction/figures/output).

### `src/preprocessing.py`

[`src/preprocessing.py`](/C:/Users/Emilio/PycharmProjects/ML_BankChurningPrediction/src/preprocessing.py) handles the feature-engineering and preprocessing setup used before modeling.

Its responsibilities are:

- remove excluded columns
- separate features and target
- define categorical and numerical feature groups
- one-hot encode categorical variables
- scale numerical variables
- save the processed CSV used for training
- create a train/test split summary for inspection

The feature-engineering summary is saved to [`results/feature_engineering_summary.txt`](/C:/Users/Emilio/PycharmProjects/ML_BankChurningPrediction/results/feature_engineering_summary.txt).

### `src/models.py`

[`src/models.py`](/C:/Users/Emilio/PycharmProjects/ML_BankChurningPrediction/src/models.py) contains the machine-learning workflow.

It includes:

- the logistic regression pipeline
- the random forest pipeline
- the shared train/test split helper
- a standalone workflow for logistic regression
- a standalone workflow for random forest
- a combined workflow that runs both models together

This file is where model training and evaluation logic lives.

### `src/evaluate.py`

[`src/evaluate.py`](/C:/Users/Emilio/PycharmProjects/ML_BankChurningPrediction/src/evaluate.py) evaluates a trained model on the test set.

It returns:

- confusion matrix
- classification report
- ROC-AUC

### `run_logistic_regression.py`

[`run_logistic_regression.py`](/C:/Users/Emilio/PycharmProjects/ML_BankChurningPrediction/run_logistic_regression.py) runs only the logistic regression workflow and saves the output to [`results/logistic_regression_metrics.txt`](/C:/Users/Emilio/PycharmProjects/ML_BankChurningPrediction/results/logistic_regression_metrics.txt).

### `run_random_forest.py`

[`run_random_forest.py`](/C:/Users/Emilio/PycharmProjects/ML_BankChurningPrediction/run_random_forest.py) runs only the random forest workflow and saves the output to [`results/random_forest_metrics.txt`](/C:/Users/Emilio/PycharmProjects/ML_BankChurningPrediction/results/random_forest_metrics.txt).

## Notebooks

The notebooks mirror the main stages of the project and are intended to make the workflow easier to follow interactively.

### `notebooks/eda_analysis.ipynb`

[`notebooks/eda_analysis.ipynb`](/C:/Users/Emilio/PycharmProjects/ML_BankChurningPrediction/notebooks/eda_analysis.ipynb) walks through the text-based EDA logic and basic churn-related findings.

### `notebooks/eda_visuals.ipynb`

[`notebooks/eda_visuals.ipynb`](/C:/Users/Emilio/PycharmProjects/ML_BankChurningPrediction/notebooks/eda_visuals.ipynb) walks through how the project visuals are generated.

### `notebooks/feature_engineering.ipynb`

[`notebooks/feature_engineering.ipynb`](/C:/Users/Emilio/PycharmProjects/ML_BankChurningPrediction/notebooks/feature_engineering.ipynb) shows how the dataset is cleaned for modeling, how excluded columns are removed, and how preprocessing is prepared.

### `notebooks/logistic_regression.ipynb`

[`notebooks/logistic_regression.ipynb`](/C:/Users/Emilio/PycharmProjects/ML_BankChurningPrediction/notebooks/logistic_regression.ipynb) focuses only on the logistic regression classifier. It shows how the processed data is loaded, how the pipeline is fit, and how the model is evaluated.

### `notebooks/random_forest.ipynb`

[`notebooks/random_forest.ipynb`](/C:/Users/Emilio/PycharmProjects/ML_BankChurningPrediction/notebooks/random_forest.ipynb) focuses only on the random forest classifier. It follows the same processed-data workflow while using the random forest model instead of logistic regression.

## How to run

### Run the full workflow

```powershell
.\.venv\Scripts\python.exe main.py
```

### Run only logistic regression

```powershell
.\.venv\Scripts\python.exe run_logistic_regression.py
```

### Run only random forest

```powershell
.\.venv\Scripts\python.exe run_random_forest.py
```

## Outputs you should expect

### Text outputs in `results/`

- `analysis_report.txt`
- `feature_engineering_summary.txt`
- `model_metrics.txt`
- `logistic_regression_metrics.txt`
- `random_forest_metrics.txt`

### Visual outputs in `figures/output/`

- `churn_distribution.svg`
- `geography_churned.svg`
- `age_vs_churn.svg`
- `is_active_member_vs_churn.svg`
