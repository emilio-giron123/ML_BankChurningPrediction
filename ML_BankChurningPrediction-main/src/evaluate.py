import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score


def evaluate_model(model, X_test, y_test):
    # Use the trained model to predict the final churn class (`0` or `1`) for
    # every row in the held-out test set.
    # Generate the predicted class labels for the test set.
    y_pred = model.predict(X_test)

    # Ask the model for class probabilities as well.
    # For binary classification, `predict_proba` returns two columns:
    # - column 0: probability of class 0 (retained)
    # - column 1: probability of class 1 (churned)
    #
    # ROC-AUC should be computed from probability scores, not from the final
    # hard class labels, so the code keeps the probability of the positive
    # class (`Exited = 1`).
    # Generate the predicted probability for the positive class (`Exited = 1`).
    # ROC-AUC uses probability scores rather than hard class labels.
    y_prob = model.predict_proba(X_test)[:, 1]

    # Collect the most important evaluation outputs in one dictionary:
    # - confusion matrix: raw count of correct / incorrect predictions
    # - classification report: precision, recall, F1-score, and support
    # - ROC-AUC: threshold-independent ranking performance
    #
    # Returning a dictionary keeps the function simple for both the combined
    # workflow and the standalone model scripts.
    # Bundle the main evaluation outputs in one dictionary so the caller can
    # print, save, or compare them later.
    results = {
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
    }

    # Return the finished metrics bundle to the caller.
    return results


def run_kfold_cv(model, X, y, n_splits=5, scoring="roc_auc"):
    """
    Runs stratified k-fold cross-validation on a given model and dataset.

    Using StratifiedKFold instead of plain KFold ensures that every fold
    preserves the same churn / non-churn class ratio as the full dataset.
    This is especially important here because the dataset is imbalanced
    (~20% churn), so a random split could produce a fold with very few
    positive examples.

    Parameters
    ----------
    model : sklearn Pipeline
        A fitted or unfitted sklearn pipeline.  `cross_val_score` re-fits
        the model on each training fold internally, so the model does not
        need to be pre-trained before calling this function.
    X : pd.DataFrame
        The full feature matrix (all rows, before the train/test split).
    y : pd.Series
        The full target vector matching `X`.
    n_splits : int
        Number of folds.  Defaults to 5.
    scoring : str
        Scoring metric passed to `cross_val_score`.  Defaults to "roc_auc"
        so the result is directly comparable to the held-out test ROC-AUC.

    Returns
    -------
    dict with keys:
        scores      – array of per-fold scores
        mean        – mean across all folds
        std         – standard deviation across all folds
        n_splits    – number of folds used
        scoring     – scoring metric used
    """
    # Build the cross-validation splitter.
    # `shuffle=True` randomises which rows end up in which fold so the result
    # is not accidentally influenced by the original row ordering in the CSV.
    # `random_state=42` makes the shuffle reproducible.
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Run cross-validation.  `cross_val_score` clones the model, fits it on
    # each training fold, and scores it on the matching held-out fold.
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)

    # Collect the per-fold scores and summary statistics so the caller can
    # print or save them without having to re-compute anything.
    return {
        "scores": scores,
        "mean": scores.mean(),
        "std": scores.std(),
        "n_splits": n_splits,
        "scoring": scoring,
    }


def run_threshold_analysis(model, X_test, y_test, thresholds=None):
    """
    Evaluates precision, recall, and F1-score at multiple probability
    thresholds instead of the default 0.5 cut-off.

    Background
    ----------
    sklearn classifiers use 0.5 as the default decision threshold: a customer
    is predicted to churn if `predict_proba(X)[:, 1] >= 0.5`.  This default
    is not always optimal, especially for imbalanced datasets.

    Lowering the threshold catches more actual churners (higher recall) at the
    cost of more false positives (lower precision).  Raising the threshold does
    the opposite.  Threshold analysis lets the project find the cut-off that
    best matches the business objective — for a bank, missing a churner is
    usually more costly than incorrectly flagging a loyal customer.

    Parameters
    ----------
    model : sklearn Pipeline
        A trained sklearn pipeline with a `predict_proba` method.
    X_test : pd.DataFrame
        Held-out feature matrix.
    y_test : pd.Series
        True churn labels for the test rows.
    thresholds : list of float, optional
        Probability cut-offs to evaluate.  Defaults to
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7] if not provided.

    Returns
    -------
    list of dicts, one per threshold, each containing:
        threshold   – the probability cut-off used
        precision   – precision score at that threshold
        recall      – recall score at that threshold
        f1          – F1-score at that threshold
        n_predicted – number of customers flagged as churners
    """
    if thresholds is None:
        # Cover a range from aggressive (0.1) to conservative (0.7) so the
        # trade-off between precision and recall is clearly visible.
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

    # Get the predicted churn probability for each test customer.
    # Column 1 is the probability that `Exited = 1` (churned).
    y_prob = model.predict_proba(X_test)[:, 1]

    results = []
    for threshold in thresholds:
        # Convert continuous probabilities to binary predictions using the
        # current threshold instead of the default 0.5.
        y_pred = (y_prob >= threshold).astype(int)

        # Compute precision, recall, and F1 with `zero_division=0` so the
        # function does not raise an error if a threshold produces zero
        # positive predictions (possible at very high thresholds).
        results.append(
            {
                "threshold": threshold,
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "recall": recall_score(y_test, y_pred, zero_division=0),
                "f1": f1_score(y_test, y_pred, zero_division=0),
                "n_predicted": int(y_pred.sum()),
            }
        )

    return results
