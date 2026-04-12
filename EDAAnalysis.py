import csv
import statistics
from collections import Counter, defaultdict
from pathlib import Path


# Centralized dataset location so the rest of the script can load the file
# without repeating path-building logic.
DATA_PATH = Path("data") / "Customer-Churn-Records.csv"
# This is the churn label used when checking which columns may leak the answer.
TARGET_COLUMN = "Exited"
IDENTIFIER_COLUMNS = {"RowNumber", "CustomerId", "Surname"}


def load_rows(path: Path) -> list[dict[str, str]]:
    # Open the CSV file at the given path.
    # `utf-8-sig` is used so the script can safely handle files that may contain
    # a UTF-8 byte order mark at the start.
    with path.open(newline="", encoding="utf-8-sig") as csv_file:
        # `csv.DictReader` reads the header row first and then creates one
        # dictionary per data row:
        # - dictionary keys   -> column names from the CSV header
        # - dictionary values -> the raw cell values as strings
        #
        # Wrapping the reader with `list(...)` loads the entire dataset into
        # memory so the rest of the script can reuse the rows many times for
        # different summaries without reopening the file.
        return list(csv.DictReader(csv_file))


def infer_dtype(values: list[str]) -> str:
    # Remove empty strings first.
    # This prevents missing values from making a mostly numeric column look like
    # text when we infer the type.
    non_empty = [value for value in values if value != ""]

    # If nothing remains after removing blanks, there is not enough information
    # to infer a useful type.
    if not non_empty:
        return "unknown"

    # Step 1: try to treat every value as an integer.
    # We test each value one by one using `int(value)`.
    # If every conversion succeeds, the whole column is considered integer data.
    try:
        for value in non_empty:
            int(value)
        return "int"
    except ValueError:
        # If even one value cannot be turned into an integer, Python raises
        # `ValueError`, and we move on to the next broader numeric type.
        pass

    # Step 2: if integer parsing failed, try float parsing.
    # This catches decimal numbers such as balances or salaries.
    try:
        for value in non_empty:
            float(value)
        return "float"
    except ValueError:
        # Step 3: if values are neither all ints nor all floats, treat the
        # column as text/categorical data.
        return "string"


def is_numeric(dtype: str) -> bool:
    # Used to decide whether a column should receive numeric or categorical
    # summary statistics.
    return dtype in {"int", "float"}


def numeric_summary(values: list[str]) -> str:
    # Convert every non-empty string value to a float so mathematical operations
    # can be performed consistently, even if the original values were integers.
    numbers = [float(value) for value in values if value != ""]

    # Count tells us how many valid numeric observations are available.
    count = len(numbers)
    # Mean gives the average value in the column.
    mean = sum(numbers) / count
    # Minimum and maximum show the full observed range.
    minimum = min(numbers)
    maximum = max(numbers)
    # Median gives the middle value, which is useful when the data is skewed.
    median = statistics.median(numbers)
    # Standard deviation measures how spread out the values are.
    # If there is only one data point, stdev would fail, so we return 0.0.
    std_dev = statistics.stdev(numbers) if count > 1 else 0.0

    # Return one formatted summary string so the caller can print it directly.
    return (
        f"count={count}, mean={mean:.4f}, std={std_dev:.4f}, "
        f"min={minimum:.4f}, median={median:.4f}, max={maximum:.4f}"
    )


def categorical_summary(values: list[str]) -> str:
    # Remove empty values so missing entries do not affect frequency counts.
    non_empty = [value for value in values if value != ""]
    # `Counter` counts how many times each category appears.
    counts = Counter(non_empty)
    # `most_common(1)` returns the single most frequent category and its count.
    top_value, top_frequency = counts.most_common(1)[0]

    # Return a compact description of the column:
    # - total non-empty values
    # - number of unique categories
    # - most common category
    # - frequency of that category
    return (
        f"count={len(non_empty)}, unique={len(counts)}, "
        f"top={top_value}, freq={top_frequency}"
    )


def exited_rate_by_group(rows: list[dict[str, str]], column: str) -> dict[str, tuple[int, float]]:
    # Create a dictionary that will store two running values for each category:
    # - position 0 -> how many rows belong to that category
    # - position 1 -> how many of those rows have Exited = 1
    #
    # `defaultdict(lambda: [0, 0])` automatically creates `[0, 0]` the first
    # time a new category value is seen.
    grouped = defaultdict(lambda: [0, 0])

    # Loop through every row in the dataset.
    for row in rows:
        # Increase the total number of customers for this category.
        grouped[row[column]][0] += 1
        # Add the churn label to the second position.
        # Because Exited is stored as "0" or "1", converting it to int lets us
        # accumulate the number of churned customers directly.
        grouped[row[column]][1] += int(row[TARGET_COLUMN])

    rates = {}
    # Convert the raw counts into churn rates.
    for value, (count, exited_count) in grouped.items():
        # Store both:
        # - the number of rows in that category
        # - the fraction that churned
        rates[value] = (count, exited_count / count)
    return rates


def identify_useless_columns(rows: list[dict[str, str]], columns: list[str]) -> list[str]:
    # This function looks for columns that should probably not be used as model
    # features because they are likely to hurt generalization or leak the target.
    row_count = len(rows)
    useless = []

    # Check each column one at a time.
    for column in columns:
        # Build a set of all values in the column so duplicates collapse into one
        # entry. The length of the set is the number of unique values.
        unique_count = len({row[column] for row in rows})

        # If every row has a different value, the column behaves like an ID.
        # ID-like columns help a model memorize training rows rather than learn
        # repeatable patterns about churn.
        if unique_count == row_count:
            useless.append(
                f"{column}: unique for every row, so it behaves like an identifier rather than a predictive feature."
            )
            continue

        # Even if surname is not unique in every row, it is still a weak feature
        # for real-world churn prediction because it is a personal label with
        # many possible values and little stable business meaning.
        if column == "Surname":
            useless.append(
                "Surname: high-cardinality personal identifier with weak generalizable signal for churn."
            )

    # Check whether the `Complain` column is effectively revealing the answer.
    # We compare churn rates for Complain=0 and Complain=1.
    complain_rates = exited_rate_by_group(rows, "Complain")
    if set(complain_rates) == {"0", "1"}:
        # Measure how far apart the two churn rates are.
        complain_gap = abs(complain_rates["1"][1] - complain_rates["0"][1])
        # If the gap is extremely large, the column is likely target leakage.
        if complain_gap > 0.99:
            useless.append(
                "Complain: near-direct leakage of the target because complaint status almost perfectly predicts churn."
            )

    return useless


def pearson_correlation(x_values: list[float], y_values: list[float]) -> float:
    # Pearson correlation measures the strength and direction of a linear
    # relationship between two numeric variables.
    # Result range:
    # - close to  1 -> strong positive relationship
    # - close to -1 -> strong negative relationship
    # - close to  0 -> weak linear relationship

    # Compute the average of each variable.
    mean_x = sum(x_values) / len(x_values)
    mean_y = sum(y_values) / len(y_values)

    # Numerator:
    # Sum how much X and Y move together around their means.
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(x_values, y_values))
    # Denominator:
    # Scale by the variability of X and Y so the result stays between -1 and 1.
    sum_sq_x = sum((x - mean_x) ** 2 for x in x_values)
    sum_sq_y = sum((y - mean_y) ** 2 for y in y_values)
    denominator = (sum_sq_x * sum_sq_y) ** 0.5

    # If one variable never changes, correlation is undefined. Returning 0.0 is
    # a practical fallback for this lightweight analysis script.
    if denominator == 0:
        return 0.0
    return numerator / denominator


def numeric_feature_correlations(
    rows: list[dict[str, str]], columns: list[str], dtypes: dict[str, str]
) -> list[tuple[str, float, float, float]]:
    # Convert the target column once so it can be reused for every feature.
    target_values = [float(row[TARGET_COLUMN]) for row in rows]
    numeric_results = []

    # Evaluate each column and keep only numeric, non-identifier features.
    for column in columns:
        if column == TARGET_COLUMN or column in IDENTIFIER_COLUMNS or not is_numeric(dtypes[column]):
            continue

        # Convert the current feature column to floats so correlation can be
        # calculated against the numeric target.
        values = [float(row[column]) for row in rows]
        # Compute the linear relationship between this feature and churn.
        correlation = pearson_correlation(values, target_values)

        # Split the feature values into two groups:
        # - customers who churned
        # - customers who did not churn
        churned_values = [float(row[column]) for row in rows if row[TARGET_COLUMN] == "1"]
        non_churned_values = [float(row[column]) for row in rows if row[TARGET_COLUMN] == "0"]

        # Compute simple averages for both groups so the user can easily see how
        # churned and non-churned customers differ.
        mean_churned = sum(churned_values) / len(churned_values)
        mean_non_churned = sum(non_churned_values) / len(non_churned_values)

        # Save one result tuple per feature.
        numeric_results.append((column, correlation, mean_churned, mean_non_churned))

    # Sort by absolute correlation so both strong positive and strong negative
    # relationships appear near the top.
    return sorted(numeric_results, key=lambda item: abs(item[1]), reverse=True)


def categorical_feature_differences(
    rows: list[dict[str, str]], columns: list[str], dtypes: dict[str, str]
) -> list[tuple[str, float, list[tuple[str, int, float]]]]:
    categorical_results = []

    # Evaluate each non-numeric, non-identifier feature.
    for column in columns:
        if column == TARGET_COLUMN or column in IDENTIFIER_COLUMNS or is_numeric(dtypes[column]):
            continue

        # Get churn rates for each category value in the current feature.
        rates = exited_rate_by_group(rows, column)
        # Sort categories from highest churn rate to lowest churn rate.
        sorted_rates = sorted(rates.items(), key=lambda item: item[1][1], reverse=True)
        # Measure how strongly this feature separates churn behavior by comparing
        # the highest and lowest category churn rates.
        strength = sorted_rates[0][1][1] - sorted_rates[-1][1][1]
        # Flatten the dictionary data into a list of tuples that is easier to
        # print later.
        details = [(value, count, rate) for value, (count, rate) in sorted_rates]
        categorical_results.append((column, strength, details))

    # Stronger churn-rate gaps are listed first.
    return sorted(categorical_results, key=lambda item: item[1], reverse=True)


def main() -> None:
    # Step 1: load the dataset into memory once.
    # Every later report section uses these same rows.
    rows = load_rows(DATA_PATH)

    # Step 2: extract basic structural information from the loaded data.
    # `columns` comes from the keys in the first row dictionary.
    columns = list(rows[0].keys())
    row_count = len(rows)
    column_count = len(columns)

    # Step 3: reorganize the data by columns instead of rows.
    # This makes it easier to compute per-column summaries later.
    column_values = {column: [row[column] for row in rows] for column in columns}

    # Step 4: infer a simple data type label for each column so the script knows
    # whether to use numeric or categorical summaries.
    dtypes = {column: infer_dtype(values) for column, values in column_values.items()}

    # Step 5: print the first five raw rows so the user can visually inspect the
    # dataset contents and formatting.
    print("First 5 rows")
    for row in rows[:5]:
        print(row)

    # Step 6: print the overall dimensions of the dataset.
    print("\nShape")
    print((row_count, column_count))

    # Step 7: print all column names.
    print("\nColumn names")
    print(columns)

    # Step 8: print the inferred type for each column.
    print("\nData types")
    for column in columns:
        print(f"{column}: {dtypes[column]}")

    # Step 9: print summary statistics for every column.
    print("\nSummary statistics")
    # Numeric columns get statistical summaries.
    # Text/categorical columns get frequency-based summaries.
    for column in columns:
        if is_numeric(dtypes[column]):
            print(f"{column}: {numeric_summary(column_values[column])}")
        else:
            print(f"{column}: {categorical_summary(column_values[column])}")

    # Step 10: print how many unique values exist in each column.
    # This helps spot IDs, low-cardinality categories, and high-cardinality text.
    print("\nUnique values per column")
    for column in columns:
        unique_count = len(set(column_values[column]))
        print(f"{column}: {unique_count}")

    # Step 11: identify columns that are likely poor features.
    print("\nLikely useless columns")
    for reason in identify_useless_columns(rows, columns):
        print(f"- {reason}")

    # Step 12: rank numeric features by how strongly they relate to churn.
    print("\nStrongest predictors of churn")
    print("Numeric features by correlation with Exited")
    for column, correlation, mean_churned, mean_non_churned in numeric_feature_correlations(rows, columns, dtypes)[:8]:
        print(
            f"{column}: correlation={correlation:.4f}, "
            f"mean_churned={mean_churned:.4f}, mean_non_churned={mean_non_churned:.4f}"
        )

    # Step 13: rank categorical features by the gap between their highest and
    # lowest churn-rate categories.
    print("\nCategorical features by churn-rate difference")
    for column, strength, details in categorical_feature_differences(rows, columns, dtypes):
        print(f"{column}: strongest churn-rate gap={strength:.4f}")
        # Only print the first few category rows to keep the output readable.
        for value, count, rate in details[:5]:
            print(f"  {value}: count={count}, exited_rate={rate:.4f}")

    # Step 14: print the churn rate for the `Complain` feature directly because
    # it is suspiciously predictive and likely leaks the target.
    complain_rates = exited_rate_by_group(rows, "Complain")
    print("\nLeakage check")
    for value in sorted(complain_rates):
        count, exited_rate = complain_rates[value]
        print(f"Complain={value}: count={count}, exited_rate={exited_rate:.4f}")


if __name__ == "__main__":
    main()
