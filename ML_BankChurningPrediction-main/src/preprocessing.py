from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import DATA_PATH, PROCESSED_DATA_PATH, RESULTS_DIR, TARGET_COLUMN

# Categorical columns that need one-hot encoding before modeling.
CATEGORICAL_COLS = ["Geography", "Gender", "Card Type"]

# Numeric columns that should be standardized before feeding them to models
# that are sensitive to feature scale.
NUMERICAL_COLS = [
    "CreditScore",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "EstimatedSalary",
    "Satisfaction Score",
    "Point Earned",
]

# Columns intentionally removed from the modeling dataset because they are
# identifiers or leakage-prone fields that should not be used for training.
DROP_COLS = ["RowNumber", "CustomerId", "Surname", "Complain"]


def prepare_features(df):
    """
    Drops unnecessary columns and separates features and target.
    """
    # Drop only the columns that are actually present in the input dataframe.
    # This makes the function work for both the raw dataset and the already
    # processed dataset written to `data/processed/`.
    removable_cols = [column for column in DROP_COLS if column in df.columns]
    df = df.drop(columns=removable_cols)

    # Split the cleaned dataframe into:
    # - X: feature columns used for training
    # - y: target column the models try to predict
    X = df.drop(TARGET_COLUMN, axis=1)
    y = df[TARGET_COLUMN]

    return X, y


def build_preprocessor():
    """
    Builds preprocessing pipeline:
    - One-hot encoding for categorical features
    - Scaling for numerical features
    """
    categorical_transformer = OneHotEncoder(drop="first")

    numerical_transformer = StandardScaler()

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, CATEGORICAL_COLS),
            ("num", numerical_transformer, NUMERICAL_COLS),
        ]
    )

    return preprocessor


def build_logistic_pipeline(preprocessor):
    # Create a reusable pipeline that applies preprocessing first and then
    # trains a logistic regression classifier on the transformed features.
    from sklearn.linear_model import LogisticRegression

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )


def build_random_forest_pipeline(preprocessor):
    # Create a reusable pipeline that applies preprocessing first and then
    # trains a random forest classifier on the transformed features.
    from sklearn.ensemble import RandomForestClassifier

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(n_estimators=100, random_state=42)),
        ]
    )


def save_processed_dataset(df: pd.DataFrame, output_path=PROCESSED_DATA_PATH):
    """
    Writes a cleaned version of the dataset to `data/processed/`.

    The saved file removes the excluded columns so downstream models train from
    the processed dataset instead of the original raw CSV.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    removable_cols = [column for column in DROP_COLS if column in df.columns]
    processed_df = df.drop(columns=removable_cols).copy()
    processed_df.to_csv(output_path, index=False)

    return processed_df, output_path


def run_feature_engineering_workflow(
    data_path=DATA_PATH,
    processed_output_path=PROCESSED_DATA_PATH,
    results_dir=RESULTS_DIR,
):
    """
    Runs the same feature-engineering flow demonstrated in the notebook:
    - load the raw dataset with pandas
    - save a processed CSV with excluded columns removed
    - drop unused columns and split X / y
    - build the preprocessing transformer
    - fit-transform the feature matrix
    - create a stratified train/test split

    Returns a dictionary with the main intermediate objects so the workflow can
    be reused by other code if needed.
    """
    # Load the raw dataset from the configured input file.
    df = pd.read_csv(data_path)

    # Persist the cleaned dataset that downstream models should train on.
    processed_df, saved_path = save_processed_dataset(df, processed_output_path)

    # Build the modeling inputs from the processed dataframe.
    X, y = prepare_features(processed_df)
    preprocessor = build_preprocessor()

    # Fit the preprocessing transformer on the full feature matrix so the code
    # can report the transformed matrix shape after encoding/scaling.
    X_transformed = preprocessor.fit_transform(X)

    # Create a stratified train/test split for later modeling stages.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Print a compact summary for the terminal run.
    print("\nFeature engineering")
    print(f"Loaded dataframe shape: {df.shape}")
    print(f"Processed dataframe shape: {processed_df.shape}")
    print(f"Processed CSV saved to: {saved_path}")
    print(f"Feature matrix shape before preprocessing: {X.shape}")
    print(f"Target vector shape: {y.shape}")
    print(f"Transformed feature matrix shape: {X_transformed.shape}")
    print(f"Train feature shape: {X_train.shape}")
    print(f"Test feature shape: {X_test.shape}")
    print(f"Train target shape: {y_train.shape}")
    print(f"Test target shape: {y_test.shape}")

    # Save the same summary to the results directory so it is kept as an
    # artifact of the run.
    results_dir.mkdir(parents=True, exist_ok=True)
    summary_path = results_dir / "feature_engineering_summary.txt"
    summary_path.write_text(
        "\n".join(
            [
                "Feature engineering",
                f"Loaded dataframe shape: {df.shape}",
                f"Processed dataframe shape: {processed_df.shape}",
                f"Processed CSV saved to: {saved_path}",
                f"Feature matrix shape before preprocessing: {X.shape}",
                f"Target vector shape: {y.shape}",
                f"Transformed feature matrix shape: {X_transformed.shape}",
                f"Train feature shape: {X_train.shape}",
                f"Test feature shape: {X_test.shape}",
                f"Train target shape: {y_train.shape}",
                f"Test target shape: {y_test.shape}",
            ]
        ),
        encoding="utf-8",
    )

    return {
        "df": df,
        "processed_df": processed_df,
        "processed_path": saved_path,
        "X": X,
        "y": y,
        "preprocessor": preprocessor,
        "X_transformed": X_transformed,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "summary_path": summary_path,
    }
