import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score


def evaluate_model(model, X_test, y_test):
    # Generate the final churn prediction (`0` or `1`) for each row in the
    # held-out test set.
    y_pred = model.predict(X_test)

    # Generate the predicted probability of churn for each row.
    # `predict_proba(... )[:, 1]` keeps only the probability that `Exited = 1`.
    # ROC-AUC should be based on probabilities rather than hard class labels.
    y_prob = model.predict_proba(X_test)[:, 1]

    # Bundle the key evaluation outputs into one dictionary so the caller can
    # print them, save them, or compare them with another model later.
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    results = {
        "y_pred": y_pred,
        "y_prob": y_prob,
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "fpr": fpr,
        "tpr": tpr,
        "roc_thresholds": thresholds,
    }

    # Return the completed metrics bundle to the caller.
    return results


def run_kfold_cv(model, X, y, n_splits=5, scoring="roc_auc"):
    # Build a stratified k-fold splitter so each fold keeps a similar churn /
    # non-churn ratio to the full dataset.
    #
    # `shuffle=True` reduces the chance that CSV row order affects the folds.
    # `random_state=42` keeps the split reproducible across repeated runs.
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Run cross-validation across the full dataset.
    # `cross_val_score` will:
    # - clone the model for each fold
    # - fit on that fold's training subset
    # - score on that fold's validation subset
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)

    # Return both the individual fold scores and the overall mean / standard
    # deviation so later code can summarize model stability.
    return {
        "scores": scores,
        "mean": scores.mean(),
        "std": scores.std(),
        "n_splits": n_splits,
        "scoring": scoring,
    }


def run_threshold_analysis(model, X_test, y_test, thresholds=None):
    if thresholds is None:
        # Test a range of cutoffs from aggressive (`0.1`) to conservative
        # (`0.7`) so the precision / recall trade-off is easy to inspect.
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

    # Generate the churn probability for each held-out customer.
    # Column 1 is the probability that `Exited = 1`.
    y_prob = model.predict_proba(X_test)[:, 1]

    results = []
    for threshold in thresholds:
        # Convert probabilities into final churn predictions using the current
        # threshold instead of the default `0.5`.
        y_pred = (y_prob >= threshold).astype(int)

        # Compute the key classification metrics at this threshold.
        # `zero_division=0` prevents errors if no rows are predicted as churned
        # at a very strict threshold.
        results.append(
            {
                "threshold": threshold,
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "recall": recall_score(y_test, y_pred, zero_division=0),
                "f1": f1_score(y_test, y_pred, zero_division=0),
                "n_predicted": int(y_pred.sum()),
            }
        )

    # Return one result row per tested threshold.
    return results
