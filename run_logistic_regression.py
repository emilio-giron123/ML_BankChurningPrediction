from src.models import run_logistic_regression_workflow


def main() -> None:
    # This standalone script is the dedicated entry point for the logistic
    # regression model.
    #
    # Running this file does not execute the random forest workflow.
    # It only:
    # 1. loads the processed dataset
    # 2. prepares the shared preprocessing and train/test split
    # 3. trains the logistic regression classifier
    # 4. evaluates that classifier
    # 5. writes the metrics file to `results/logistic_regression_metrics.txt`
    results = run_logistic_regression_workflow()

    # Print the saved artifact path so the user can immediately find the
    # metrics file produced by this standalone run.
    print(f"Saved: {results['metrics_path']}")


if __name__ == "__main__":
    # Only run the workflow when this file is executed directly.
    main()
