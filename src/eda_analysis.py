import statistics
from collections import Counter, defaultdict
from pathlib import Path
from io import StringIO
from contextlib import redirect_stdout

from src.config import (
    DATA_PATH,
    EXCLUDED_PREDICTOR_COLUMNS,
    IDENTIFIER_COLUMNS,
    RESULTS_DIR,
    TARGET_COLUMN,
)
from src.data_loader import load_rows


def infer_dtype(values: list[str]) -> str:
    # Remove blank strings before inferring the type so missing values do not
    # force an otherwise numeric column to be treated as text.
    non_empty = [value for value in values if value != ""]

    # If no usable values remain, there is not enough information to infer a
    # meaningful type for the column.
    if not non_empty:
        return "unknown"

    # Try integer parsing first because it is the most specific numeric type.
    # If every non-empty value can be converted with `int(...)`, the column is
    # treated as integer data.
    try:
        for value in non_empty:
            int(value)
        return "int"
    except ValueError:
        # If any value fails integer parsing, move on to float parsing.
        pass

    # Try float parsing next. This handles decimal-valued columns such as
    # balances and salaries.
    try:
        for value in non_empty:
            float(value)
        return "float"
    except ValueError:
        # If the values are neither all ints nor all floats, treat the column
        # as string/categorical data.
        return "string"


def is_numeric(dtype: str) -> bool:
    # Return True only for the two numeric labels used by this project.
    # This helper keeps the type check consistent anywhere the code needs to
    # decide between numeric and categorical logic.
    return dtype in {"int", "float"}


def numeric_summary(values: list[str]) -> str:
    # Convert non-empty strings to floats so one set of math operations can
    # handle both integer-looking and decimal-valued columns.
    numbers = [float(value) for value in values if value != ""]

    # Count how many usable numeric values exist in the column.
    count = len(numbers)
    # Compute the average of the numeric values.
    mean = sum(numbers) / count
    # Capture the observed lower and upper bounds.
    minimum = min(numbers)
    maximum = max(numbers)
    # Compute the first quartile (25th percentile) using the inclusive method.
    q1 = statistics.quantiles(numbers, n=4, method="inclusive")[0]
    # Compute the middle value of the sorted numbers.
    median = statistics.median(numbers)
    # Compute the third quartile (75th percentile).
    q3 = statistics.quantiles(numbers, n=4, method="inclusive")[2]
    # The interquartile range shows the spread of the middle 50% of the data.
    iqr = q3 - q1
    # Standard deviation measures how spread out the values are around the mean.
    # If there is only one value, stdev would be undefined, so use 0.0.
    std_dev = statistics.stdev(numbers) if count > 1 else 0.0

    # Return a formatted summary string so the calling code can print the
    # statistics without needing to know how each value was calculated.
    return (
        f"count={count}, mean={mean:.4f}, std={std_dev:.4f}, "
        f"min={minimum:.4f}, Q1={q1:.4f}, median={median:.4f}, "
        f"Q3={q3:.4f}, IQR={iqr:.4f}, max={maximum:.4f}"
    )


def categorical_summary(values: list[str]) -> str:
    # Ignore blank values so missing entries do not affect category counts.
    non_empty = [value for value in values if value != ""]
    # Count how often each unique category appears in the column.
    counts = Counter(non_empty)
    # Identify the single most common category and how many times it appears.
    top_value, top_frequency = counts.most_common(1)[0]

    # Return a compact description of the categorical column:
    # - total non-empty values
    # - number of unique categories
    # - most common category
    # - frequency of that category
    return (
        f"count={len(non_empty)}, unique={len(counts)}, "
        f"top={top_value}, freq={top_frequency}"
    )


def exited_rate_by_group(
    rows: list[dict[str, str]], column: str
) -> dict[str, tuple[int, float]]:
    # Use a defaultdict so every new category automatically starts with:
    # - total row count = 0
    # - churned row count = 0
    grouped = defaultdict(lambda: [0, 0])

    # Walk through every row and accumulate totals for the requested column.
    for row in rows:
        # Increment how many rows belong to the current category.
        grouped[row[column]][0] += 1
        # Add the churn label (`0` or `1`) to the churned counter.
        # Converting the string to int allows direct numeric accumulation.
        grouped[row[column]][1] += int(row[TARGET_COLUMN])

    rates = {}
    # Convert raw counts into a friendlier output structure containing:
    # - the number of rows in the category
    # - the churn rate for that category
    for value, (count, exited_count) in grouped.items():
        rates[value] = (count, exited_count / count)
    return rates


def identify_useless_columns(
    rows: list[dict[str, str]], columns: list[str]
) -> list[str]:
    # `row_count` is used to spot columns where every row has a unique value.
    # Columns like that usually act as identifiers rather than useful features.
    row_count = len(rows)
    useless = []

    # Check each column one by one.
    for column in columns:
        # Build a set of the column values to count the number of unique values.
        unique_count = len({row[column] for row in rows})

        # If every row has a distinct value, the column behaves like an ID.
        # That usually makes it poor for modeling because the model learns row
        # identity instead of repeatable churn patterns.
        if unique_count == row_count:
            useless.append(
                f"{column}: unique for every row, so it behaves like an identifier rather than a predictive feature."
            )
            continue

        # `Surname` is also treated as a weak feature because it is a personal,
        # high-cardinality label with little stable business meaning.
        if column == "Surname":
            useless.append(
                "Surname: high-cardinality personal identifier with weak generalizable signal for churn."
            )

    # `Complain` is checked separately because it may not be unique, but it can
    # still be unusable if it almost directly reveals the target.
    # Only run the complaint leakage check if the processed dataset still
    # contains that column. The processed training file removes it.
    if "Complain" in columns:
        complain_rates = exited_rate_by_group(rows, "Complain")
        if set(complain_rates) == {"0", "1"}:
            # Measure how far apart the churn rates are for complaint status 0 vs 1.
            complain_gap = abs(complain_rates["1"][1] - complain_rates["0"][1])
            # If the gap is extremely large, the feature is likely target leakage.
            if complain_gap > 0.99:
                useless.append(
                    "Complain: near-direct leakage of the target because complaint status almost perfectly predicts churn."
                )

    return useless


def pearson_correlation(x_values: list[float], y_values: list[float]) -> float:
    # Compute the average value of each input variable.
    mean_x = sum(x_values) / len(x_values)
    mean_y = sum(y_values) / len(y_values)

    # Numerator:
    # measure how much X and Y move together around their means.
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(x_values, y_values))
    # Denominator:
    # scale by the spread of X and the spread of Y so the final result stays
    # between -1 and 1.
    sum_sq_x = sum((x - mean_x) ** 2 for x in x_values)
    sum_sq_y = sum((y - mean_y) ** 2 for y in y_values)
    denominator = (sum_sq_x * sum_sq_y) ** 0.5

    # If either variable has no variance, the correlation is undefined.
    # Returning 0.0 is a safe fallback for this lightweight report.
    if denominator == 0:
        return 0.0
    return numerator / denominator


def numeric_feature_correlations(
    rows: list[dict[str, str]], columns: list[str], dtypes: dict[str, str]
) -> list[tuple[str, float, float, float]]:
    # Convert the target column once so it can be reused for every feature.
    target_values = [float(row[TARGET_COLUMN]) for row in rows]
    numeric_results = []

    # Evaluate each feature column and keep only valid numeric predictors.
    for column in columns:
        if (
            column == TARGET_COLUMN
            or column in IDENTIFIER_COLUMNS
            or column in EXCLUDED_PREDICTOR_COLUMNS
            or not is_numeric(dtypes[column])
        ):
            continue

        # Convert the current feature to floats for numeric analysis.
        values = [float(row[column]) for row in rows]
        # Measure the linear relationship between this feature and the churn label.
        correlation = pearson_correlation(values, target_values)

        # Split the feature values into the churned and non-churned groups so
        # the report can show how the two populations differ in plain language.
        churned_values = [float(row[column]) for row in rows if row[TARGET_COLUMN] == "1"]
        non_churned_values = [float(row[column]) for row in rows if row[TARGET_COLUMN] == "0"]

        # Calculate simple group averages for easier interpretation.
        mean_churned = sum(churned_values) / len(churned_values)
        mean_non_churned = sum(non_churned_values) / len(non_churned_values)

        # Store the result tuple for later ranking.
        numeric_results.append((column, correlation, mean_churned, mean_non_churned))

    # Sort by absolute correlation so strong negative and strong positive
    # relationships both appear near the top.
    return sorted(numeric_results, key=lambda item: abs(item[1]), reverse=True)


def categorical_feature_differences(
    rows: list[dict[str, str]], columns: list[str], dtypes: dict[str, str]
) -> list[tuple[str, float, list[tuple[str, int, float]]]]:
    categorical_results = []

    # Evaluate each valid categorical predictor.
    for column in columns:
        if (
            column == TARGET_COLUMN
            or column in IDENTIFIER_COLUMNS
            or column in EXCLUDED_PREDICTOR_COLUMNS
            or is_numeric(dtypes[column])
        ):
            continue

        # Calculate the churn rate for each category in the feature.
        rates = exited_rate_by_group(rows, column)
        # Sort categories from highest churn rate to lowest churn rate.
        sorted_rates = sorted(rates.items(), key=lambda item: item[1][1], reverse=True)
        # Use the difference between the highest and lowest churn rate as a
        # simple measure of how strongly the feature separates churn behavior.
        strength = sorted_rates[0][1][1] - sorted_rates[-1][1][1]
        # Flatten the nested dictionary into printable tuples.
        details = [(value, count, rate) for value, (count, rate) in sorted_rates]
        categorical_results.append((column, strength, details))

    # Sort the features so the ones with the biggest churn-rate separation
    # appear first in the report.
    return sorted(categorical_results, key=lambda item: item[1], reverse=True)


def build_column_values(rows: list[dict[str, str]], columns: list[str]) -> dict[str, list[str]]:
    # Rearrange the data from row-oriented form:
    #   row -> {"Age": "42", "Gender": "Female", ...}
    # into column-oriented form:
    #   "Age" -> ["42", "41", ...]
    #
    # This makes it easier to summarize one column at a time.
    return {column: [row[column] for row in rows] for column in columns}


def infer_all_dtypes(column_values: dict[str, list[str]]) -> dict[str, str]:
    # Run the single-column type inference helper across every column and return
    # a dictionary mapping each column name to its inferred data type.
    return {column: infer_dtype(values) for column, values in column_values.items()}


def _print_analysis_report(path: Path = DATA_PATH) -> None:
    # Load the dataset from the configured path.
    rows = load_rows(path)
    # Extract the column names from the first row dictionary.
    columns = list(rows[0].keys())
    # Capture the overall shape for later reporting.
    row_count = len(rows)
    column_count = len(columns)

    # Precompute a column-oriented view of the data so later report sections do
    # not have to rebuild the same structure repeatedly.
    column_values = build_column_values(rows, columns)
    # Infer one simple data type label for every column.
    dtypes = infer_all_dtypes(column_values)

    # Show a quick preview of the raw records.
    print("First 5 rows")
    for row in rows[:5]:
        print(row)

    # Show the total number of rows and columns.
    print("\nShape")
    print((row_count, column_count))

    # List all dataset columns in order.
    print("\nColumn names")
    print(columns)

    # Print the inferred type for each column.
    print("\nData types")
    for column in columns:
        print(f"{column}: {dtypes[column]}")

    # Print a summary for each column.
    # Numeric columns get distribution statistics.
    # Non-numeric columns get frequency-based summaries.
    print("\nSummary statistics")
    for column in columns:
        if is_numeric(dtypes[column]):
            print(f"{column}: {numeric_summary(column_values[column])}")
        else:
            print(f"{column}: {categorical_summary(column_values[column])}")

    # Count how many distinct values appear in each column.
    # This helps identify binary flags, low-cardinality categories, and IDs.
    print("\nUnique values per column")
    for column in columns:
        unique_count = len(set(column_values[column]))
        print(f"{column}: {unique_count}")

    # Highlight columns that are likely poor modeling features.
    print("\nUseless columns")
    for reason in identify_useless_columns(rows, columns):
        print(f"- {reason}")

    # Rank numeric predictors by correlation with the churn target.
    print("\nStrongest predictors of churn")
    print("Numeric features by correlation with Exited")
    for column, correlation, mean_churned, mean_non_churned in numeric_feature_correlations(
        rows, columns, dtypes
    )[:8]:
        print(
            f"{column}: correlation={correlation:.4f}, "
            f"mean_churned={mean_churned:.4f}, mean_non_churned={mean_non_churned:.4f}"
        )

    # Rank categorical predictors by how different their category churn rates are.
    print("\nCategorical features by churn-rate difference")
    for column, strength, details in categorical_feature_differences(rows, columns, dtypes):
        print(f"{column}: strongest churn-rate gap={strength:.4f}")
        # Only print the first few categories per feature to keep the output readable.
        for value, count, rate in details[:5]:
            print(f"  {value}: count={count}, exited_rate={rate:.4f}")

    # Print the complaint leakage relationship only when the current dataset
    # still includes that column. The processed dataset intentionally removes it.
    if "Complain" in columns:
        complain_rates = exited_rate_by_group(rows, "Complain")
        print("\nLeakage check")
        for value in sorted(complain_rates):
            count, exited_rate = complain_rates[value]
            print(f"Complain={value}: count={count}, exited_rate={exited_rate:.4f}")


def run_analysis_report(path: Path = DATA_PATH, results_dir: Path = RESULTS_DIR) -> Path:
    # Runs the analysis report, prints to the terminal, and saves into `results/analysis_report.txt`.
    buffer = StringIO()
    with redirect_stdout(buffer):
        _print_analysis_report(path)

    report_text = buffer.getvalue()
    print(report_text, end="")

    results_dir.mkdir(parents=True, exist_ok=True)
    report_path = results_dir / "analysis_report.txt"
    report_path.write_text(report_text, encoding="utf-8")
    return report_path


if __name__ == "__main__":
    # Allow the report module to be run directly for standalone analysis.
    run_analysis_report()
