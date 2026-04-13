from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

import pandas as pd

from src.config import PROCESSED_DATA_PATH, RESULTS_DIR
from src.evaluate import evaluate_model
from src.preprocessing import build_preprocessor, prepare_features


def build_logistic_model(preprocessor):
    # Build a single sklearn Pipeline object so the same preprocessing steps
    # used during training are also applied automatically during prediction.
    #
    # The pipeline runs in this order:
    # 1. `preprocessor` transforms the raw dataframe columns
    #    - categorical columns are one-hot encoded
    #    - numeric columns are scaled
    # 2. `LogisticRegression` learns the relationship between those processed
    #    features and the binary churn target (`Exited`)
    #
    # Keeping preprocessing and modeling in one pipeline prevents training /
    # prediction mismatches and makes the workflow easier to reuse.
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )


def build_random_forest_model(preprocessor):
    # Build a single sklearn Pipeline object for the random forest workflow.
    #
    # This follows the same pattern as the logistic regression pipeline:
    # 1. apply the shared preprocessing logic to the dataframe columns
    # 2. fit a RandomForestClassifier on the transformed features
    #
    # Using the same preprocessing structure for both models keeps the project
    # consistent and makes model comparisons more fair.
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(n_estimators=100, random_state=42)),
        ]
    )


def _train_test_data(data_path=PROCESSED_DATA_PATH):
    # Load the processed dataset created by the feature-engineering step.
    # This keeps model training focused on the cleaned file in
    # `data/processed/` instead of the original raw CSV.
    df = pd.read_csv(data_path)

    # Split the dataframe into:
    # - X: all predictor columns
    # - y: the churn target column
    #
    # `prepare_features` also removes columns that should never be used for
    # training, such as identifiers and the leakage-prone `Complain` field.
    X, y = prepare_features(df)

    # Build the shared preprocessing transformer that knows how to encode
    # categorical columns and scale numeric columns.
    preprocessor = build_preprocessor()

    # Split the data into training and testing subsets.
    #
    # Step by step:
    # 1. reserve 20% of the rows for final evaluation
    # 2. keep 80% for model fitting
    # 3. use `random_state=42` so repeated runs are reproducible
    # 4. use `stratify=y` so both splits preserve the original churn / non-churn
    #    balance as closely as possible
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Return all important intermediate objects in one dictionary so the
    # standalone model workflows and the combined workflow can reuse the same
    # prepared data without duplicating setup code.
    return {
        "df": df,
        "X": X,
        "y": y,
        "preprocessor": preprocessor,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }


def run_logistic_regression_workflow(
    data_path=PROCESSED_DATA_PATH, results_dir=RESULTS_DIR
):
    # Prepare the shared dataset split and preprocessing objects.
    # This makes the standalone logistic regression run behave the same way as
    # the combined modeling workflow.
    split_data = _train_test_data(data_path)

    # Build the logistic regression pipeline that wraps preprocessing plus
    # classifier training in one object.
    model = build_logistic_model(split_data["preprocessor"])

    # Fit the pipeline on the training data.
    # During this step:
    # 1. the preprocessor learns how to transform the training columns
    # 2. the logistic regression model learns the churn relationship from the
    #    transformed training features
    model.fit(split_data["X_train"], split_data["y_train"])

    # Evaluate the trained model on the held-out test set so performance is
    # measured on unseen data.
    results = evaluate_model(model, split_data["X_test"], split_data["y_test"])

    # Print the key metrics to the terminal for quick inspection.
    print("\nLogistic Regression")
    print(results["classification_report"])
    print(f"ROC-AUC: {results['roc_auc']:.4f}")

    # Make sure the results directory exists before writing the metrics file.
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = results_dir / "logistic_regression_metrics.txt"

    # Save the same evaluation summary to disk so the logistic regression run
    # can be reviewed later without rerunning the model.
    metrics_path.write_text(
        "\n".join(
            [
                "Logistic Regression",
                "",
                results["classification_report"],
                f"ROC-AUC: {results['roc_auc']:.4f}",
            ]
        ),
        encoding="utf-8",
    )

    # Return both the trained model and the prepared data objects so other code
    # can inspect or reuse them if needed.
    return {
        "model": model,
        "results": results,
        "metrics_path": metrics_path,
        **split_data,
    }


def run_random_forest_workflow(
    data_path=PROCESSED_DATA_PATH, results_dir=RESULTS_DIR
):
    # Prepare the shared dataset split and preprocessing objects.
    # This keeps the standalone random forest run aligned with the combined
    # modeling workflow.
    split_data = _train_test_data(data_path)

    # Build the random forest pipeline that bundles preprocessing and model
    # training together.
    model = build_random_forest_model(split_data["preprocessor"])

    # Fit the full pipeline on the training data.
    # The preprocessor transforms the columns first, then the random forest
    # learns patterns that separate churned from retained customers.
    model.fit(split_data["X_train"], split_data["y_train"])

    # Evaluate the trained model on the held-out test set.
    results = evaluate_model(model, split_data["X_test"], split_data["y_test"])

    # Print the main metrics to the terminal for quick inspection.
    print("\nRandom Forest")
    print(results["classification_report"])
    print(f"ROC-AUC: {results['roc_auc']:.4f}")

    # Make sure the results directory exists before writing the metrics file.
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = results_dir / "random_forest_metrics.txt"

    # Save the evaluation summary so the standalone random forest run has its
    # own persistent output file.
    metrics_path.write_text(
        "\n".join(
            [
                "Random Forest",
                "",
                results["classification_report"],
                f"ROC-AUC: {results['roc_auc']:.4f}",
            ]
        ),
        encoding="utf-8",
    )

    # Return both the trained model and the prepared data objects so callers
    # can inspect the workflow output programmatically.
    return {
        "model": model,
        "results": results,
        "metrics_path": metrics_path,
        **split_data,
    }


def run_modeling_workflow(data_path=PROCESSED_DATA_PATH, results_dir=RESULTS_DIR):
    """
    Run the combined modeling workflow used by `main.py`.

    Step by step, this function:
    - loads the processed dataset
    - prepares the shared train/test split
    - builds the logistic regression pipeline
    - builds the random forest pipeline
    - fits both models on the training partition
    - evaluates both models on the held-out test partition
    - prints the metrics
    - saves the combined metrics summary to `results/model_metrics.txt`

    Returns a dictionary containing the trained models and evaluation results.
    """
    split_data = _train_test_data(data_path)
    X_train = split_data["X_train"]
    X_test = split_data["X_test"]
    y_train = split_data["y_train"]
    y_test = split_data["y_test"]

    # Build both model pipelines from the same preprocessing definition so the
    # comparison is based on the model choice rather than mismatched data prep.
    log_model = build_logistic_model(split_data["preprocessor"])
    rf_model = build_random_forest_model(split_data["preprocessor"])

    # Fit both models on the training partition.
    log_model.fit(X_train, y_train)
    rf_model.fit(X_train, y_train)

    # Evaluate both trained models on the same held-out test set so the reported
    # performance numbers are directly comparable.
    log_results = evaluate_model(log_model, X_test, y_test)
    rf_results = evaluate_model(rf_model, X_test, y_test)

    print("\nModeling")
    print("Logistic Regression:")
    print(log_results["classification_report"])
    print(f"ROC-AUC: {log_results['roc_auc']:.4f}")

    print("\nRandom Forest:")
    print(rf_results["classification_report"])
    print(f"ROC-AUC: {rf_results['roc_auc']:.4f}")

    # Save the combined model metrics to the results directory so they remain
    # available even after the terminal session ends.
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = results_dir / "model_metrics.txt"
    metrics_path.write_text(
        "\n".join(
            [
                "Modeling",
                "",
                "Logistic Regression:",
                log_results["classification_report"],
                f"ROC-AUC: {log_results['roc_auc']:.4f}",
                "",
                "Random Forest:",
                rf_results["classification_report"],
                f"ROC-AUC: {rf_results['roc_auc']:.4f}",
            ]
        ),
        encoding="utf-8",
    )

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "log_model": log_model,
        "rf_model": rf_model,
        "log_results": log_results,
        "rf_results": rf_results,
        "metrics_path": metrics_path,
        **split_data,
    }


def run_comparison_workflow(data_path=PROCESSED_DATA_PATH, results_dir=RESULTS_DIR):
    """
    Run the full model comparison workflow.

    This function builds on the basic modeling workflow by adding:
    - 5-fold stratified cross-validation for both models
    - Threshold analysis for both models
    - A side-by-side summary table saved to results/

    Returns a dictionary containing the trained models, evaluation results,
    cross-validation results, and threshold analysis results.
    """
    from src.evaluate import run_kfold_cv, run_threshold_analysis

    # Reuse the shared data preparation so the comparison workflow uses
    # exactly the same train/test split as the individual model workflows.
    split_data = _train_test_data(data_path)
    X_train = split_data["X_train"]
    X_test  = split_data["X_test"]
    y_train = split_data["y_train"]
    y_test  = split_data["y_test"]
    X       = split_data["X"]
    y       = split_data["y"]

    # Build and train both pipelines from the same shared preprocessor so
    # any difference in results comes only from the model, not from data prep.
    log_model = build_logistic_model(split_data["preprocessor"])
    rf_model  = build_random_forest_model(split_data["preprocessor"])

    log_model.fit(X_train, y_train)
    rf_model.fit(X_train, y_train)

    # ── Hold-out test metrics ────────────────────────────────────────────────
    log_results = evaluate_model(log_model, X_test, y_test)
    rf_results  = evaluate_model(rf_model,  X_test, y_test)

    # ── K-Fold Cross-Validation ──────────────────────────────────────────────
    # Pass the full feature matrix (X) and target (y), not just the test split,
    # so cross_val_score can create its own folds across the entire dataset.
    print("\nRunning 5-fold cross-validation for Logistic Regression...")
    log_cv = run_kfold_cv(log_model, X, y, n_splits=5, scoring="roc_auc")

    print("Running 5-fold cross-validation for Random Forest...")
    rf_cv  = run_kfold_cv(rf_model,  X, y, n_splits=5, scoring="roc_auc")

    # ── Threshold Analysis ───────────────────────────────────────────────────
    # Evaluate a range of probability thresholds on the held-out test set to
    # understand how precision and recall trade off for each model.
    log_threshold = run_threshold_analysis(log_model, X_test, y_test)
    rf_threshold  = run_threshold_analysis(rf_model,  X_test, y_test)

    # ── Print comparison summary ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 60)

    header = f"{'Metric':<22} {'Log. Regression':>18} {'Random Forest':>16}"
    print(header)
    print("-" * 60)

    # Derive scalar metrics from the classification report strings by
    # re-running predict so we have direct access to the numbers.
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    import numpy as np

    for model_obj, label, results in [
        (log_model, "Logistic Regression", log_results),
        (rf_model,  "Random Forest",       rf_results),
    ]:
        y_pred = model_obj.predict(X_test)
        results["accuracy"]  = accuracy_score(y_test, y_pred)
        results["precision"] = precision_score(y_test, y_pred)
        results["recall"]    = recall_score(y_test, y_pred)
        results["f1"]        = f1_score(y_test, y_pred)

    metrics = [
        ("Accuracy",        "accuracy"),
        ("Precision",       "precision"),
        ("Recall",          "recall"),
        ("F1-Score",        "f1"),
        ("ROC-AUC (test)",  "roc_auc"),
    ]
    for label, key in metrics:
        print(f"{label:<22} {log_results[key]:>18.4f} {rf_results[key]:>16.4f}")

    print("-" * 60)
    print(f"{'CV ROC-AUC Mean':<22} {log_cv['mean']:>18.4f} {rf_cv['mean']:>16.4f}")
    print(f"{'CV ROC-AUC Std':<22} {log_cv['std']:>18.4f} {rf_cv['std']:>16.4f}")

    print("\n--- Logistic Regression: Cross-Validation Fold Scores ---")
    for i, score in enumerate(log_cv["scores"], 1):
        print(f"  Fold {i}: {score:.4f}")

    print("\n--- Random Forest: Cross-Validation Fold Scores ---")
    for i, score in enumerate(rf_cv["scores"], 1):
        print(f"  Fold {i}: {score:.4f}")

    print("\n--- Logistic Regression: Threshold Analysis ---")
    print(f"  {'Threshold':>10} {'Precision':>10} {'Recall':>8} {'F1':>8} {'# Predicted':>12}")
    for row in log_threshold:
        print(f"  {row['threshold']:>10.1f} {row['precision']:>10.4f} "
              f"{row['recall']:>8.4f} {row['f1']:>8.4f} {row['n_predicted']:>12}")

    print("\n--- Random Forest: Threshold Analysis ---")
    print(f"  {'Threshold':>10} {'Precision':>10} {'Recall':>8} {'F1':>8} {'# Predicted':>12}")
    for row in rf_threshold:
        print(f"  {row['threshold']:>10.1f} {row['precision']:>10.4f} "
              f"{row['recall']:>8.4f} {row['f1']:>8.4f} {row['n_predicted']:>12}")

    # ── Save results ─────────────────────────────────────────────────────────
    results_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = results_dir / "model_comparison.txt"

    lines = [
        "=" * 60,
        "MODEL COMPARISON SUMMARY",
        "=" * 60,
        header,
        "-" * 60,
    ]
    for label, key in metrics:
        lines.append(f"{label:<22} {log_results[key]:>18.4f} {rf_results[key]:>16.4f}")
    lines += [
        "-" * 60,
        f"{'CV ROC-AUC Mean':<22} {log_cv['mean']:>18.4f} {rf_cv['mean']:>16.4f}",
        f"{'CV ROC-AUC Std':<22} {log_cv['std']:>18.4f} {rf_cv['std']:>16.4f}",
        "",
        "--- Logistic Regression: Cross-Validation Fold Scores ---",
    ]
    for i, score in enumerate(log_cv["scores"], 1):
        lines.append(f"  Fold {i}: {score:.4f}")
    lines += [
        "",
        "--- Random Forest: Cross-Validation Fold Scores ---",
    ]
    for i, score in enumerate(rf_cv["scores"], 1):
        lines.append(f"  Fold {i}: {score:.4f}")
    lines += [
        "",
        "--- Logistic Regression: Threshold Analysis ---",
        f"  {'Threshold':>10} {'Precision':>10} {'Recall':>8} {'F1':>8} {'# Predicted':>12}",
    ]
    for row in log_threshold:
        lines.append(
            f"  {row['threshold']:>10.1f} {row['precision']:>10.4f} "
            f"{row['recall']:>8.4f} {row['f1']:>8.4f} {row['n_predicted']:>12}"
        )
    lines += [
        "",
        "--- Random Forest: Threshold Analysis ---",
        f"  {'Threshold':>10} {'Precision':>10} {'Recall':>8} {'F1':>8} {'# Predicted':>12}",
    ]
    for row in rf_threshold:
        lines.append(
            f"  {row['threshold']:>10.1f} {row['precision']:>10.4f} "
            f"{row['recall']:>8.4f} {row['f1']:>8.4f} {row['n_predicted']:>12}"
        )

    comparison_path.write_text("\n".join(lines), encoding="utf-8")

    return {
        "log_model":       log_model,
        "rf_model":        rf_model,
        "log_results":     log_results,
        "rf_results":      rf_results,
        "log_cv":          log_cv,
        "rf_cv":           rf_cv,
        "log_threshold":   log_threshold,
        "rf_threshold":    rf_threshold,
        "comparison_path": comparison_path,
        **split_data,
    }
