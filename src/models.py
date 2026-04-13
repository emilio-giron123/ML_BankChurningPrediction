from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

import pandas as pd

from src.config import PROCESSED_DATA_PATH, RESULTS_DIR
from src.evaluate import evaluate_model, run_kfold_cv, run_threshold_analysis
from src.model_visuals import (
    plot_confusion_matrix_analysis,
    plot_cv_roc_auc,
    plot_model_performance_comparison,
    plot_random_forest_feature_importance,
    plot_roc_curve_analysis,
    plot_threshold_vs_metrics,
)
from src.preprocessing import build_preprocessor, prepare_features


def build_logistic_model(preprocessor):
    # Build one sklearn Pipeline that combines preprocessing and logistic
    # regression in a single reusable workflow.
    #
    # The steps run in order:
    # - `preprocessor` transforms the raw dataframe columns
    # - `classifier` fits logistic regression on the transformed features
    #
    # Keeping both steps in one pipeline ensures the exact same preprocessing
    # logic is applied during both training and prediction.
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )


def build_random_forest_model(preprocessor):
    # Build one sklearn Pipeline that combines preprocessing and random forest
    # training in a single reusable workflow.
    #
    # This follows the same structure as logistic regression so both models are
    # trained from the same feature-preparation logic.
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(n_estimators=100, random_state=42)),
        ]
    )


def _train_test_data(data_path=PROCESSED_DATA_PATH):
    # Load the processed dataset created by the feature-engineering step.
    # This keeps all model training focused on the cleaned file in
    # `data/processed/` instead of the original raw dataset.
    df = pd.read_csv(data_path)

    # Split the dataframe into:
    # - `X`: all predictor columns
    # - `y`: the churn target column
    #
    # `prepare_features(...)` also removes columns that should not be used for
    # training, such as identifiers and leakage-prone fields.
    X, y = prepare_features(df)

    # Build the shared preprocessing transformer that handles:
    # - one-hot encoding for categorical features
    # - scaling for numeric features
    preprocessor = build_preprocessor()

    # Split the data into training and testing subsets.
    #
    # This reserves 20% of the data for final evaluation and keeps 80% for
    # training. `stratify=y` preserves the churn / non-churn balance.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Return the prepared objects in one dictionary so multiple workflows can
    # reuse the same setup code without duplicating it.
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


def _format_threshold_lines(threshold_results):
    # Convert the threshold-analysis dictionaries into fixed-width text lines
    # so they can be printed to the terminal and saved to text files.
    return [
        (
            f"{row['threshold']:>10.1f} {row['precision']:>10.4f} "
            f"{row['recall']:>8.4f} {row['f1']:>8.4f} {row['n_predicted']:>12}"
        )
        for row in threshold_results
    ]


def _format_cv_scores(scores):
    # Convert numpy scalar values into plain formatted strings so saved text
    # outputs show clean numeric values instead of `np.float64(...)`.
    return [f"{float(score):.4f}" for score in scores]


def run_logistic_regression_workflow(
    data_path=PROCESSED_DATA_PATH, results_dir=RESULTS_DIR
):
    # Prepare the shared dataset split and preprocessing objects so the
    # standalone logistic regression run uses the same setup as the full
    # comparison workflow.
    split_data = _train_test_data(data_path)

    # Build the logistic regression pipeline from the shared preprocessor.
    model = build_logistic_model(split_data["preprocessor"])

    # Fit the pipeline on the training subset.
    # During this step the preprocessor learns the training transformations and
    # the classifier learns how those transformed features relate to churn.
    model.fit(split_data["X_train"], split_data["y_train"])

    # Evaluate the trained model on the held-out test set so the reported
    # metrics reflect performance on unseen customers.
    results = evaluate_model(model, split_data["X_test"], split_data["y_test"])

    # Run 5-fold cross-validation on the full dataset to measure how stable
    # the model's ROC-AUC is across multiple train/validation splits.
    cv_results = run_kfold_cv(model, split_data["X"], split_data["y"], n_splits=5)

    # Run threshold analysis on the held-out test set so the user can inspect
    # the precision / recall trade-off at multiple probability cutoffs.
    threshold_results = run_threshold_analysis(
        model, split_data["X_test"], split_data["y_test"]
    )
    cv_score_strings = _format_cv_scores(cv_results["scores"])
    threshold_lines = _format_threshold_lines(threshold_results)

    # Print the main hold-out metrics first for quick inspection.
    print("\nLogistic Regression")
    print(results["classification_report"])
    print(f"ROC-AUC: {results['roc_auc']:.4f}")

    # Print the cross-validation summary so the user can see the average
    # performance and variation across folds.
    print("5-Fold CV ROC-AUC")
    print(f"Scores: {cv_score_strings}")
    print(f"Mean: {cv_results['mean']:.4f}")
    print(f"Std: {cv_results['std']:.4f}")

    # Print the threshold-analysis table so the user can compare precision,
    # recall, and F1-score at different cutoffs.
    print("Threshold Analysis")
    print(f"{'Threshold':>10} {'Precision':>10} {'Recall':>8} {'F1':>8} {'# Predicted':>12}")
    for line in threshold_lines:
        print(line)

    # Create the results directory before writing the standalone metrics file.
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = results_dir / "logistic_regression_metrics.txt"

    # Save the hold-out metrics, cross-validation summary, and threshold
    # analysis to disk so this standalone run is self-contained.
    metrics_path.write_text(
        "\n".join(
            [
                "Logistic Regression",
                "",
                results["classification_report"],
                f"ROC-AUC: {results['roc_auc']:.4f}",
                "",
                "5-Fold CV ROC-AUC",
                f"Scores: {cv_score_strings}",
                f"Mean: {cv_results['mean']:.4f}",
                f"Std: {cv_results['std']:.4f}",
                "",
                "Threshold Analysis",
                f"{'Threshold':>10} {'Precision':>10} {'Recall':>8} {'F1':>8} {'# Predicted':>12}",
                *threshold_lines,
            ]
        ),
        encoding="utf-8",
    )

    # Return the trained model, the prepared data objects, and the extended
    # evaluation results so other code can inspect or reuse them.
    return {
        "model": model,
        "results": results,
        "cv_results": cv_results,
        "threshold_results": threshold_results,
        "metrics_path": metrics_path,
        **split_data,
    }


def run_random_forest_workflow(
    data_path=PROCESSED_DATA_PATH, results_dir=RESULTS_DIR
):
    # Prepare the shared dataset split and preprocessing objects so the
    # standalone random forest run uses the same setup as the full comparison
    # workflow.
    split_data = _train_test_data(data_path)

    # Build the random forest pipeline from the shared preprocessor.
    model = build_random_forest_model(split_data["preprocessor"])

    # Fit the pipeline on the training subset.
    # During this step the preprocessor transforms the training features and
    # the random forest learns non-linear churn patterns from those features.
    model.fit(split_data["X_train"], split_data["y_train"])

    # Evaluate the trained model on the held-out test set so the reported
    # metrics reflect performance on unseen customers.
    results = evaluate_model(model, split_data["X_test"], split_data["y_test"])

    # Run 5-fold cross-validation on the full dataset to measure how stable
    # the model's ROC-AUC is across multiple train/validation splits.
    cv_results = run_kfold_cv(model, split_data["X"], split_data["y"], n_splits=5)

    # Run threshold analysis on the held-out test set so the user can inspect
    # the precision / recall trade-off at multiple probability cutoffs.
    threshold_results = run_threshold_analysis(
        model, split_data["X_test"], split_data["y_test"]
    )
    cv_score_strings = _format_cv_scores(cv_results["scores"])
    threshold_lines = _format_threshold_lines(threshold_results)

    # Print the main hold-out metrics first for quick inspection.
    print("\nRandom Forest")
    print(results["classification_report"])
    print(f"ROC-AUC: {results['roc_auc']:.4f}")

    # Print the cross-validation summary so the user can see the average
    # performance and variation across folds.
    print("5-Fold CV ROC-AUC")
    print(f"Scores: {cv_score_strings}")
    print(f"Mean: {cv_results['mean']:.4f}")
    print(f"Std: {cv_results['std']:.4f}")

    # Print the threshold-analysis table so the user can compare precision,
    # recall, and F1-score at different cutoffs.
    print("Threshold Analysis")
    print(f"{'Threshold':>10} {'Precision':>10} {'Recall':>8} {'F1':>8} {'# Predicted':>12}")
    for line in threshold_lines:
        print(line)

    # Create the results directory before writing the standalone metrics file.
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = results_dir / "random_forest_metrics.txt"
    feature_importance_path = plot_random_forest_feature_importance(model, results_dir)

    # Save the hold-out metrics, cross-validation summary, and threshold
    # analysis to disk so this standalone run is self-contained.
    metrics_path.write_text(
        "\n".join(
            [
                "Random Forest",
                "",
                results["classification_report"],
                f"ROC-AUC: {results['roc_auc']:.4f}",
                "",
                "5-Fold CV ROC-AUC",
                f"Scores: {cv_score_strings}",
                f"Mean: {cv_results['mean']:.4f}",
                f"Std: {cv_results['std']:.4f}",
                "",
                "Threshold Analysis",
                f"{'Threshold':>10} {'Precision':>10} {'Recall':>8} {'F1':>8} {'# Predicted':>12}",
                *threshold_lines,
                "",
                f"Feature importance chart: {feature_importance_path}",
            ]
        ),
        encoding="utf-8",
    )

    # Return the trained model, the prepared data objects, and the extended
    # evaluation results so other code can inspect or reuse them.
    return {
        "model": model,
        "results": results,
        "cv_results": cv_results,
        "threshold_results": threshold_results,
        "metrics_path": metrics_path,
        "feature_importance_path": feature_importance_path,
        **split_data,
    }


def run_modeling_workflow(data_path=PROCESSED_DATA_PATH, results_dir=RESULTS_DIR):
    # Run the basic side-by-side modeling workflow.
    #
    # This function:
    # - loads the processed dataset
    # - prepares the shared train/test split
    # - trains logistic regression and random forest
    # - evaluates both on the held-out test set
    # - saves the basic metrics summary to `results/model_metrics.txt`
    split_data = _train_test_data(data_path)
    X_train = split_data["X_train"]
    X_test = split_data["X_test"]
    y_train = split_data["y_train"]
    y_test = split_data["y_test"]

    # Build both pipelines from the same preprocessor so the comparison is
    # based on model choice rather than mismatched feature preparation.
    log_model = build_logistic_model(split_data["preprocessor"])
    rf_model = build_random_forest_model(split_data["preprocessor"])

    # Fit both models on the training partition.
    log_model.fit(X_train, y_train)
    rf_model.fit(X_train, y_train)

    # Evaluate both trained models on the same held-out test set so the
    # reported performance numbers are directly comparable.
    log_results = evaluate_model(log_model, X_test, y_test)
    rf_results = evaluate_model(rf_model, X_test, y_test)

    print("\nModeling")
    print("Logistic Regression:")
    print(log_results["classification_report"])
    print(f"ROC-AUC: {log_results['roc_auc']:.4f}")

    print("\nRandom Forest:")
    print(rf_results["classification_report"])
    print(f"ROC-AUC: {rf_results['roc_auc']:.4f}")

    # Save the combined metrics so the basic side-by-side model output is
    # preserved outside the terminal.
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
    # Run the full side-by-side model comparison workflow.
    #
    # This extends the basic modeling workflow by adding:
    # - 5-fold stratified cross-validation for both models
    # - threshold analysis for both models
    # - a saved side-by-side summary report in `results/model_comparison.txt`
    split_data = _train_test_data(data_path)
    X_train = split_data["X_train"]
    X_test = split_data["X_test"]
    y_train = split_data["y_train"]
    y_test = split_data["y_test"]
    X = split_data["X"]
    y = split_data["y"]

    # Build and train both pipelines from the same preprocessor so any
    # performance differences come from the model itself, not the data prep.
    log_model = build_logistic_model(split_data["preprocessor"])
    rf_model = build_random_forest_model(split_data["preprocessor"])

    log_model.fit(X_train, y_train)
    rf_model.fit(X_train, y_train)

    # Evaluate both models on the held-out test set.
    log_results = evaluate_model(log_model, X_test, y_test)
    rf_results = evaluate_model(rf_model, X_test, y_test)

    # Run 5-fold cross-validation on the full dataset for both models.
    # This uses the entire dataset because cross-validation creates its own
    # train/validation folds internally.
    print("\nRunning 5-fold cross-validation for Logistic Regression...")
    log_cv = run_kfold_cv(log_model, X, y, n_splits=5, scoring="roc_auc")

    print("Running 5-fold cross-validation for Random Forest...")
    rf_cv = run_kfold_cv(rf_model, X, y, n_splits=5, scoring="roc_auc")

    # Evaluate multiple probability cutoffs on the held-out test set so the
    # precision / recall trade-off is visible for each model.
    log_threshold = run_threshold_analysis(log_model, X_test, y_test)
    rf_threshold = run_threshold_analysis(rf_model, X_test, y_test)

    # Derive scalar metrics directly from the test-set predictions so the
    # comparison table has accuracy, precision, recall, and F1-score.
    for model_obj, results in [
        (log_model, log_results),
        (rf_model, rf_results),
    ]:
        y_pred = model_obj.predict(X_test)
        results["accuracy"] = accuracy_score(y_test, y_pred)
        results["precision"] = precision_score(y_test, y_pred)
        results["recall"] = recall_score(y_test, y_pred)
        results["f1"] = f1_score(y_test, y_pred)

    # Print a compact side-by-side comparison summary.
    print("\n" + "=" * 60)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 60)

    header = f"{'Metric':<22} {'Log. Regression':>18} {'Random Forest':>16}"
    print(header)
    print("-" * 60)

    metrics = [
        ("Accuracy", "accuracy"),
        ("Precision", "precision"),
        ("Recall", "recall"),
        ("F1-Score", "f1"),
        ("ROC-AUC (test)", "roc_auc"),
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
    for line in _format_threshold_lines(log_threshold):
        print(f"  {line}")

    print("\n--- Random Forest: Threshold Analysis ---")
    print(f"  {'Threshold':>10} {'Precision':>10} {'Recall':>8} {'F1':>8} {'# Predicted':>12}")
    for line in _format_threshold_lines(rf_threshold):
        print(f"  {line}")

    # Save the full comparison summary to the results directory.
    results_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = results_dir / "model_comparison.txt"
    performance_plot_path = plot_model_performance_comparison(
        log_results, rf_results, results_dir
    )
    confusion_matrix_path = plot_confusion_matrix_analysis(
        log_results, rf_results, results_dir
    )
    roc_curve_path = plot_roc_curve_analysis(log_results, rf_results, results_dir)
    cv_plot_path = plot_cv_roc_auc(log_cv, rf_cv, results_dir)
    threshold_plot_path = plot_threshold_vs_metrics(
        log_threshold, rf_threshold, results_dir
    )
    feature_importance_path = plot_random_forest_feature_importance(
        rf_model, results_dir
    )

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
    for line in _format_threshold_lines(log_threshold):
        lines.append(f"  {line}")
    lines += [
        "",
        "--- Random Forest: Threshold Analysis ---",
        f"  {'Threshold':>10} {'Precision':>10} {'Recall':>8} {'F1':>8} {'# Predicted':>12}",
    ]
    for line in _format_threshold_lines(rf_threshold):
        lines.append(f"  {line}")
    lines += [
        "",
        "Generated Visuals",
        f"  Model performance comparison: {performance_plot_path}",
        f"  Confusion matrix analysis: {confusion_matrix_path}",
        f"  ROC curve analysis: {roc_curve_path}",
        f"  5-fold cross-validation ROC-AUC: {cv_plot_path}",
        f"  Metrics vs threshold: {threshold_plot_path}",
        f"  Random forest top 12 feature importance: {feature_importance_path}",
    ]

    comparison_path.write_text("\n".join(lines), encoding="utf-8")

    # Return the trained models, comparison outputs, and prepared data objects
    # so the caller can inspect or reuse them later.
    return {
        "log_model": log_model,
        "rf_model": rf_model,
        "log_results": log_results,
        "rf_results": rf_results,
        "log_cv": log_cv,
        "rf_cv": rf_cv,
        "log_threshold": log_threshold,
        "rf_threshold": rf_threshold,
        "comparison_path": comparison_path,
        "performance_plot_path": performance_plot_path,
        "confusion_matrix_path": confusion_matrix_path,
        "roc_curve_path": roc_curve_path,
        "cv_plot_path": cv_plot_path,
        "threshold_plot_path": threshold_plot_path,
        "feature_importance_path": feature_importance_path,
        **split_data,
    }
