import csv
import statistics
from collections import Counter, defaultdict
from pathlib import Path


DATA_PATH = Path("data") / "Customer-Churn-Records.csv"
TARGET_COLUMN = "Exited"


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as csv_file:
        return list(csv.DictReader(csv_file))


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
    median = statistics.median(numbers)
    std_dev = statistics.stdev(numbers) if count > 1 else 0.0
    return (
        f"count={count}, mean={mean:.4f}, std={std_dev:.4f}, "
        f"min={minimum:.4f}, median={median:.4f}, max={maximum:.4f}"
    )


def categorical_summary(values: list[str]) -> str:
    non_empty = [value for value in values if value != ""]
    counts = Counter(non_empty)
    top_value, top_frequency = counts.most_common(1)[0]
    return (
        f"count={len(non_empty)}, unique={len(counts)}, "
        f"top={top_value}, freq={top_frequency}"
    )


def exited_rate_by_group(rows: list[dict[str, str]], column: str) -> dict[str, tuple[int, float]]:
    grouped = defaultdict(lambda: [0, 0])
    for row in rows:
        grouped[row[column]][0] += 1
        grouped[row[column]][1] += int(row[TARGET_COLUMN])

    rates = {}
    for value, (count, exited_count) in grouped.items():
        rates[value] = (count, exited_count / count)
    return rates


def identify_useless_columns(rows: list[dict[str, str]], columns: list[str]) -> list[str]:
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


def main() -> None:
    rows = load_rows(DATA_PATH)
    columns = list(rows[0].keys())
    row_count = len(rows)
    column_count = len(columns)
    column_values = {column: [row[column] for row in rows] for column in columns}
    dtypes = {column: infer_dtype(values) for column, values in column_values.items()}

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

    complain_rates = exited_rate_by_group(rows, "Complain")
    print("\nLeakage check")
    for value in sorted(complain_rates):
        count, exited_rate = complain_rates[value]
        print(f"Complain={value}: count={count}, exited_rate={exited_rate:.4f}")


if __name__ == "__main__":
    main()
