# ML Bank Churning Prediction

This project performs exploratory data analysis on a bank customer churn dataset and generates a small set of EDA visuals.

The code is organized so that:
- `main.py` is the entry point
- reusable logic lives in `src/`
- raw data lives in `data/raw/`
- generated figures are written to `figures/output/`

## Project structure

```text
ML_BankChurningPrediction/
|- data/
|  |- raw/
|  |  |- Customer-Churn-Records.csv
|  |- processed/
|- figures/
|  |- output/
|  |- EDAVisuals.py
|- notebooks/
|- results/
|- src/
|  |- __init__.py
|  |- config.py
|  |- data_loader.py
|  |- eda_analysis.py
|  |- eda_visuals.py
|- main.py
|- README.md
```

## What the program does

When you run `main.py`, the program does two things in order:

1. It prints an EDA report for the churn dataset.
2. It generates visual files and saves them inside `figures/output/`.

The report includes:
- first 5 rows
- dataset shape
- column names
- inferred data types
- summary statistics
- unique value counts
- likely useless columns
- strongest predictors of churn
- a leakage check for `Complain`

The visuals include:
- churn distribution
- geography vs churn
- age vs churn
- `IsActiveMember` vs churn

## How execution works

The full runtime flow is:

1. `main.py` starts the program.
2. `main.py` imports `run_analysis_report` from `src.eda_analysis`.
3. `main.py` imports `generate_all_eda_visuals` from `src.eda_visuals`.
4. `run_analysis_report()` loads the CSV data and prints the text-based analysis.
5. `generate_all_eda_visuals()` loads the same CSV data and creates `.svg` charts.
6. `main.py` prints the saved output paths for the generated visual files.

So `main.py` is only the coordinator. The real analysis and plotting logic lives in the `src` package.

## File-by-file explanation

### `main.py`

This is the entry point for the whole project.

It does not contain the analysis logic itself. Instead, it calls two high-level functions:
- `run_analysis_report()`
- `generate_all_eda_visuals()`

Its job is:
- start the report
- start figure generation
- print the saved figure paths

If you want to run the full project workflow, this is the file to run.

### `src/__init__.py`

This file marks `src/` as a Python package.

That allows imports such as:

```python
from src.eda_analysis import run_analysis_report
```

Without this file, Python would not treat `src` as a proper importable package.

### `src/config.py`

This file stores shared project constants in one place.

It defines:
- `PROJECT_ROOT`: the root folder of the repository
- `DATA_PATH`: the path to the raw churn CSV file
- `FIGURES_OUTPUT_DIR`: where charts should be saved
- `TARGET_COLUMN`: the target label, which is `Exited`
- `IDENTIFIER_COLUMNS`: columns that behave like identifiers and should not be treated as meaningful predictors
- `EXCLUDED_PREDICTOR_COLUMNS`: columns excluded from predictor ranking, currently `Complain`

Why this matters:
- path logic is centralized
- column rules are centralized
- other files do not need to repeat the same hardcoded values

### `src/data_loader.py`

This file is responsible for loading the CSV dataset.

It currently contains one function:

```python
load_rows(path: Path) -> list[dict[str, str]]
```

What it does:
- opens the CSV file
- reads the header row
- converts every data row into a dictionary
- returns all rows as a list

Each row looks conceptually like this:

```python
{
    "CreditScore": "619",
    "Geography": "France",
    "Age": "42",
    "Exited": "1"
}
```

This means the rest of the code works with named columns instead of raw CSV lines.

### `src/eda_analysis.py`

This file contains the text-based analysis logic.

Its main public function is:

```python
run_analysis_report()
```

This function:
- loads the dataset
- extracts column names
- builds a column-oriented view of the data
- infers simple types for each column
- prints multiple EDA sections

Important helper functions inside this file:

#### `infer_dtype(values)`

Determines whether a column should be treated as:
- `int`
- `float`
- `string`
- `unknown`

It ignores empty strings before trying conversions.

#### `is_numeric(dtype)`

Returns `True` if the inferred type is numeric.

This is used to decide whether a column gets numeric summary statistics or categorical summary statistics.

#### `numeric_summary(values)`

Builds summary statistics for numeric columns.

It reports:
- count
- mean
- standard deviation
- minimum
- Q1
- median
- Q3
- IQR
- maximum

#### `categorical_summary(values)`

Builds summary information for non-numeric columns.

It reports:
- number of non-empty values
- number of unique values
- most common value
- frequency of the most common value

#### `exited_rate_by_group(rows, column)`

Computes churn rate by category.

Examples:
- churn rate by geography
- churn rate by `IsActiveMember`
- churn rate by `Complain`

It returns both:
- group size
- exited rate

#### `identify_useless_columns(rows, columns)`

Identifies columns that are poor modeling features.

It flags:
- row-level identifiers such as `RowNumber` and `CustomerId`
- `Surname` because it is a high-cardinality personal identifier
- `Complain` because it behaves like target leakage

#### `pearson_correlation(x_values, y_values)`

Computes Pearson correlation between a numeric feature and the target.

This is used to find which numeric features are most related to churn.

#### `numeric_feature_correlations(rows, columns, dtypes)`

Ranks numeric features by the strength of their correlation with `Exited`.

It excludes:
- the target column itself
- identifier-like columns
- `Complain`

For each feature, it reports:
- correlation with churn
- mean value for churned customers
- mean value for non-churned customers

#### `categorical_feature_differences(rows, columns, dtypes)`

Ranks categorical features by how different their churn rates are across categories.

For example:
- if one geography has much higher churn than another, that feature will rank higher

#### `build_column_values(rows, columns)`

Transforms the data from row-based form into column-based form.

This makes it easier to summarize each column efficiently.

#### `infer_all_dtypes(column_values)`

Runs `infer_dtype` across all columns and returns a dictionary of inferred types.

### `src/eda_visuals.py`

This file contains the chart generation logic.

Its main public function is:

```python
generate_all_eda_visuals()
```

This function:
- creates the output directory if needed
- loads the dataset
- calls each plotting function
- returns the list of generated file paths

Important functions in this module:

#### `ensure_output_dir(output_dir)`

Creates the output folder if it does not already exist.

#### `plot_churn_distribution(rows, output_dir)`

Creates a bar chart showing:
- not churned count
- churned count

Saved as:
- `figures/output/churn_distribution.svg`

#### `plot_geography_vs_churn(rows, output_dir)`

Creates a grouped bar chart comparing churned vs not churned customers for each geography.

Saved as:
- `figures/output/geography_vs_churn.svg`

#### `plot_age_vs_churn(rows, output_dir)`

Creates a boxplot of age split by churn status.

This helps show how the age distribution differs between churned and non-churned customers.

Saved as:
- `figures/output/age_vs_churn.svg`

#### `plot_is_active_member_vs_churn(rows, output_dir)`

Creates a bar chart of churn rate for:
- inactive members
- active members

Saved as:
- `figures/output/is_active_member_vs_churn.svg`

## Data location

The current code expects the dataset here:

[`data/raw/Customer-Churn-Records.csv`](/C:/Users/Emilio/PycharmProjects/ML_BankChurningPrediction/data/raw/Customer-Churn-Records.csv)

That path is controlled by `src/config.py`.

## Generated outputs

The visual outputs are written here:

[`figures/output`](/C:/Users/Emilio/PycharmProjects/ML_BankChurningPrediction/figures/output)

Current expected files:
- `churn_distribution.svg`
- `geography_vs_churn.svg`
- `age_vs_churn.svg`
- `is_active_member_vs_churn.svg`

## How to run the project

From the project root:

```powershell
.\.venv\Scripts\python.exe main.py
```

If you want to run the modules separately:

```powershell
.\.venv\Scripts\python.exe src\eda_analysis.py
.\.venv\Scripts\python.exe src\eda_visuals.py
```

## What to expect when you run `main.py`

You should see:
- the printed EDA report in the terminal
- one `Saved: ...` line per generated figure

The final output is:
- a text summary of the dataset and churn relationships
- a set of SVG visualizations in `figures/output/`

## Summary

In simple terms:
- `main.py` runs everything
- `src/config.py` stores shared paths and constants
- `src/data_loader.py` reads the CSV
- `src/eda_analysis.py` prints the analysis report
- `src/eda_visuals.py` creates the charts

That is the full working flow of the current codebase.
