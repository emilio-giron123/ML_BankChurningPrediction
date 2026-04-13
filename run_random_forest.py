from src.models import run_random_forest_workflow


def main() -> None:
    # This standalone script is the dedicated entry point for the random forest
    # model.
    #
    # Running this file does not execute the logistic regression workflow.
    # It only:
    # 1. loads the processed dataset
    # 2. prepares the shared preprocessing and train/test split
    # 3. trains the random forest classifier
    # 4. evaluates that classifier
    # 5. writes the metrics file to `results/random_forest_metrics.txt`
    results = run_random_forest_workflow()

    # Print the saved artifact path so the user can immediately find the
    # metrics file and feature-importance chart produced by this standalone run.
    print(f"Saved: {results['metrics_path']}")
    print(f"Saved: {results['feature_importance_path']}")


if __name__ == "__main__":
    # Only run the workflow when this file is executed directly.
    main()
