from src.eda_analysis import run_analysis_report
from src.eda_visuals import generate_all_eda_visuals
from src.models import run_modeling_workflow
from src.preprocessing import run_feature_engineering_workflow


def main() -> None:
    # Run the full text-based exploratory analysis first.
    # This prints the dataset preview, summaries, churn relationships,
    # and leakage checks directly to the terminal while also saving a copy
    # of the report to the results directory.
    analysis_report_path = run_analysis_report()

    # Generate all requested EDA charts and collect the saved file paths.
    # The plotting function returns a list of Path objects pointing to
    # the SVG files created inside the figures output directory.
    paths = generate_all_eda_visuals()

    # Print each saved path so the user can immediately see where the
    # generated figure files were written.
    for path in paths:
        print(f"Saved: {path}")

    # Run the preprocessing / feature-engineering workflow that is also shown
    # in the feature engineering notebook.
    feature_results = run_feature_engineering_workflow()

    # Train and evaluate the baseline models shown in the modeling notebook,
    # using the processed dataset created by the feature-engineering step.
    modeling_results = run_modeling_workflow(data_path=feature_results["processed_path"])

    # Print the saved result artifact paths so it is obvious where the
    # non-figure outputs were written.
    print(f"Saved: {analysis_report_path}")
    print(f"Saved: {feature_results['summary_path']}")
    print(f"Saved: {modeling_results['metrics_path']}")


if __name__ == "__main__":
    # Only execute the pipeline when this file is run directly.
    # This prevents the workflow from running automatically if `main.py`
    # is imported from another module.
    main()
