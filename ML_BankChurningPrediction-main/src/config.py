from pathlib import Path

# Compute the repository root by starting from this file's location
# (`src/config.py`), moving up one directory to `src/`, and then
# moving up again to the project root folder.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Keep an explicit handle to the original raw dataset so it is still available
# if the project ever needs to rebuild the processed file from scratch.
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "Customer-Churn-Records.csv"

# Path to the processed dataset produced after feature-engineering cleanup.
# This file is the one the modeling workflow should learn from.
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "customer_churn_processed.csv"

# The project now defaults to the processed dataset for downstream work.
# EDA, visualization, and modeling should all read from this file unless a
# caller explicitly passes a different path.
DATA_PATH = PROCESSED_DATA_PATH

# Shared output directory where generated figures are saved.
FIGURES_OUTPUT_DIR = PROJECT_ROOT / "figures" / "output"

# Root directory for saved text summaries and model-result artifacts.
RESULTS_DIR = PROJECT_ROOT / "results"

# Name of the target column. This is the field the project treats as
# the churn label when calculating churn rates and predictor strength.
TARGET_COLUMN = "Exited"

# Columns that should not be treated as useful predictive features
# because they behave like identifiers rather than general patterns.
IDENTIFIER_COLUMNS = {"RowNumber", "CustomerId", "Surname"}

# Columns excluded from the "strongest predictors" ranking even though
# they may be highly related to churn. `Complain` is excluded because it
# behaves like target leakage rather than a fair predictor.
EXCLUDED_PREDICTOR_COLUMNS = {"Complain"}
