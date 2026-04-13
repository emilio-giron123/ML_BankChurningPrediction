from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


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
