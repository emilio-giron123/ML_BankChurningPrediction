from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "Customer-Churn-Records.csv"
FIGURES_OUTPUT_DIR = PROJECT_ROOT / "figures" / "output"

TARGET_COLUMN = "Exited"
IDENTIFIER_COLUMNS = {"RowNumber", "CustomerId", "Surname"}
EXCLUDED_PREDICTOR_COLUMNS = {"Complain"}