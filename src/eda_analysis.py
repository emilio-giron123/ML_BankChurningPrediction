import statistics
from collections import Counter, defaultdict
from pathlib import Path

from src.config import (
    DATA_PATH,
    EXCLUDED_PREDICTOR_COLUMNS,
    IDENTIFIER_COLUMNS,
    TARGET_COLUMN,
)
from src.data_loader import load_rows


def infer_dtype(values: list[str]) -> str:
    non_empty = [value for value in values if value != ""]
    if not non_empty:
        return "unknown"

    try:
        for value in non_empty:
            int(value)
        return "int"
    except ValueError:
        pass

    try:
        for value in non_empty:
            float(value)
        return "float"
    except ValueError:
        return "string"


def is_numeric(dtype: str) -> bool:
    return dtype in {"int", "float"}


def numeric_summary(values: list[str]) -> str:
    numbers = [float(value) for value in values if value != ""]
    count = len(numbers)
    mean = sum(numbers) / count
    minimum = min(numbers)
    maximum = max(numbers)
    q1 = statistics.quantiles(numbers, n=4, method="inclusive")[0]
    median = statistics.median(numbers)
    q3 = statistics.quantiles(numbers, n=4, method="inclusive")[2]
    iqr = q3 - q1
    std_dev = statistics.stdev(numbers) if count > 1 else 0.0

    return (
        f"count={count}, mean={mean:.4f}, std={std_dev:.4f}, "
        f"min={minimum:.4f}, Q1={q1:.4f}, median={median:.4f}, "
        f"Q3={q3:.4f}, IQR={iqr:.4f}, max={maximum:.4f}"
    )


def categorical_summary(values: list[str]) -> str:
    non_empty = [value for value in values if value != ""]
    counts = Counter(non_empty)
    top_value, top_frequency = counts.most_common(1)[0]

    return (
        f"count={len(non_empty)}, unique={len(counts)}, "
        f"top={top_value}, freq={top_frequency}"
    )


def exited_rate_by_group(
    rows: list[dict[str, str]], column: str
) -> dict[str, tuple[int, float]]:
    grouped = defaultdict(lambda: [0, 0])

    for row in rows:
        grouped[row[column]][0] += 1
        grouped[row[column]][1] += int(row[TARGET_COLUMN])

    rates = {}
    for value, (count, exited_count) in grouped.items():
        rates[value] = (count, exited_count / count)
    return rates


def identify_useless_columns(
    rows: list[dict[str, str]], columns: list[str]
) -> list[str]:
    row_count = len(rows)
    useless = []

    for column in columns:
        unique_count = len({row[column] for row in rows})

        if unique_count == row_count:
            useless.append(
                f"{column}: unique for every row, so it behaves like an identifier rather than a predictive feature."
            )
            continue

        if column == "Surname":
            useless.append(
                "Surname: high-cardinality personal identifier with weak generalizable signal for churn."
            )

    complain_rates = exited_rate_by_group(rows, "Complain")
    if set(complain_rates) == {"0", "1"}:
        complain_gap = abs(complain_rates["1"][1] - complain_rates["0"][1])
        if complain_gap > 0.99:
            useless.append(
                "Complain: near-direct leakage of the target because complaint status almost perfectly predicts churn."
            )

    return useless


def pearson_correlation(x_values: list[float], y_values: list[float]) -> float:
    mean_x = sum(x_values) / len(x_values)
    mean_y = sum(y_values) / len(y_values)

    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(x_values, y_values))
    sum_sq_x = sum((x - mean_x) ** 2 for x in x_values)
    sum_sq_y = sum((y - mean_y) ** 2 for y in y_values)
    denominator = (sum_sq_x * sum_sq_y) ** 0.5

    if denominator == 0:
        return 0.0
    return numerator / denominator


def numeric_feature_correlations(
    rows: list[dict[str, str]], columns: list[str], dtypes: dict[str, str]
) -> list[tuple[str, float, float, float]]:
    target_values = [float(row[TARGET_COLUMN]) for row in rows]
    numeric_results = []

    for column in columns:
        if (
            column == TARGET_COLUMN
            or column in IDENTIFIER_COLUMNS
            or column in EXCLUDED_PREDICTOR_COLUMNS
            or not is_numeric(dtypes[column])
        ):
            continue

        values = [float(row[column]) for row in rows]
        correlation = pearson_correlation(values, target_values)

        churned_values = [float(row[column]) for row in rows if row[TARGET_COLUMN] == "1"]
        non_churned_values = [float(row[column]) for row in rows if row[TARGET_COLUMN] == "0"]

        mean_churned = sum(churned_values) / len(churned_values)
        mean_non_churned = sum(non_churned_values) / len(non_churned_values)

        numeric_results.append((column, correlation, mean_churned, mean_non_churned))

    return sorted(numeric_results, key=lambda item: abs(item[1]), reverse=True)


def categorical_feature_differences(
    rows: list[dict[str, str]], columns: list[str], dtypes: dict[str, str]
) -> list[tuple[str, float, list[tuple[str, int, float]]]]:
    categorical_results = []

    for column in columns:
        if (
            column == TARGET_COLUMN
            or column in IDENTIFIER_COLUMNS
            or column in EXCLUDED_PREDICTOR_COLUMNS
            or is_numeric(dtypes[column])
        ):
            continue

        rates = exited_rate_by_group(rows, column)
        sorted_rates = sorted(rates.items(), key=lambda item: item[1][1], reverse=True)
        strength = sorted_rates[0][1][1] - sorted_rates[-1][1][1]
        details = [(value, count, rate) for value, (count, rate) in sorted_rates]
        categorical_results.append((column, strength, details))

    return sorted(categorical_results, key=lambda item: item[1], reverse=True)


def build_column_values(rows: list[dict[str, str]], columns: list[str]) -> dict[str, list[str]]:
    return {column: [row[column] for row in rows] for column in columns}


def infer_all_dtypes(column_values: dict[str, list[str]]) -> dict[str, str]:
    return {column: infer_dtype(values) for column, values in column_values.items()}


def run_analysis_report(path: Path = DATA_PATH) -> None:
    rows = load_rows(path)
    columns = list(rows[0].keys())
    row_count = len(rows)
    column_count = len(columns)

    column_values = build_column_values(rows, columns)
    dtypes = infer_all_dtypes(column_values)

    print("First 5 rows")
    for row in rows[:5]:
        print(row)

    print("\nShape")
    print((row_count, column_count))

    print("\nColumn names")
    print(columns)

    print("\nData types")
    for column in columns:
        print(f"{column}: {dtypes[column]}")

    print("\nSummary statistics")
    for column in columns:
        if is_numeric(dtypes[column]):
            print(f"{column}: {numeric_summary(column_values[column])}")
        else:
            print(f"{column}: {categorical_summary(column_values[column])}")

    print("\nUnique values per column")
    for column in columns:
        unique_count = len(set(column_values[column]))
        print(f"{column}: {unique_count}")

    print("\nLikely useless columns")
    for reason in identify_useless_columns(rows, columns):
        print(f"- {reason}")

    print("\nStrongest predictors of churn")
    print("Numeric features by correlation with Exited")
    for column, correlation, mean_churned, mean_non_churned in numeric_feature_correlations(
        rows, columns, dtypes
    )[:8]:
        print(
            f"{column}: correlation={correlation:.4f}, "
            f"mean_churned={mean_churned:.4f}, mean_non_churned={mean_non_churned:.4f}"
        )

    print("\nCategorical features by churn-rate difference")
    for column, strength, details in categorical_feature_differences(rows, columns, dtypes):
        print(f"{column}: strongest churn-rate gap={strength:.4f}")
        for value, count, rate in details[:5]:
            print(f"  {value}: count={count}, exited_rate={rate:.4f}")

    complain_rates = exited_rate_by_group(rows, "Complain")
    print("\nLeakage check")
    for value in sorted(complain_rates):
        count, exited_rate = complain_rates[value]
        print(f"Complain={value}: count={count}, exited_rate={exited_rate:.4f}")


if __name__ == "__main__":
    run_analysis_report()