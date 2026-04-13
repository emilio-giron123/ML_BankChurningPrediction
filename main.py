from src.eda_analysis import run_analysis_report
from src.eda_visuals import generate_all_eda_visuals
from src.models import run_comparison_workflow
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

    # Run the full model-comparison workflow once.
    # This single call now handles:
    # - training both baseline models
    # - hold-out evaluation
    # - k-fold cross-validation
    # - threshold analysis
    # - saving the comparison summary
    comparison_results = run_comparison_workflow(
        data_path=feature_results["processed_path"]
    )

    # Print the saved result artifact paths so it is obvious where the
    # non-figure outputs were written.
    print(f"Saved: {analysis_report_path}")
    print(f"Saved: {feature_results['summary_path']}")
    print(f"Saved: {comparison_results['comparison_path']}")
    print(f"Saved: {comparison_results['performance_plot_path']}")
    print(f"Saved: {comparison_results['confusion_matrix_path']}")
    print(f"Saved: {comparison_results['roc_curve_path']}")
    print(f"Saved: {comparison_results['cv_plot_path']}")
    print(f"Saved: {comparison_results['threshold_plot_path']}")
    print(f"Saved: {comparison_results['feature_importance_path']}")


if __name__ == "__main__":
    # Only execute the pipeline when this file is run directly.
    # This prevents the workflow from running automatically if `main.py`
    # is imported from another module.
    main()
