from pathlib import Path

import matplotlib.pyplot as plt

from src.config import DATA_PATH, FIGURES_OUTPUT_DIR, TARGET_COLUMN
from src.data_loader import load_rows
from src.eda_analysis import exited_rate_by_group


def ensure_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)


def plot_churn_distribution(rows: list[dict[str, str]], output_dir: Path) -> Path:
    counts = {"0": 0, "1": 0}
    for row in rows:
        counts[row[TARGET_COLUMN]] += 1

    labels = ["Not Churned", "Churned"]
    values = [counts["0"], counts["1"]]

    plt.figure(figsize=(8, 5))
    plt.bar(labels, values)
    plt.title("Churn Distribution")
    plt.xlabel("Customer Status")
    plt.ylabel("Count")
    plt.tight_layout()

    output_path = output_dir / "churn_distribution.svg"
    plt.savefig(output_path, format="svg")
    plt.close()
    return output_path


def plot_geography_vs_churn(rows: list[dict[str, str]], output_dir: Path) -> Path:
    geography_counts: dict[str, dict[str, int]] = {}
    for row in rows:
        geography = row["Geography"]
        exited = row[TARGET_COLUMN]

        if geography not in geography_counts:
            geography_counts[geography] = {"0": 0, "1": 0}
        geography_counts[geography][exited] += 1

    geographies = list(geography_counts.keys())
    not_churned = [geography_counts[g]["0"] for g in geographies]
    churned = [geography_counts[g]["1"] for g in geographies]

    x = range(len(geographies))
    width = 0.35

    plt.figure(figsize=(8, 5))
    plt.bar([i - width / 2 for i in x], not_churned, width=width, label="Not Churned")
    plt.bar([i + width / 2 for i in x], churned, width=width, label="Churned")
    plt.xticks(list(x), geographies)
    plt.title("Geography vs Churn")
    plt.xlabel("Geography")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()

    output_path = output_dir / "geography_vs_churn.svg"
    plt.savefig(output_path, format="svg")
    plt.close()
    return output_path


def plot_age_vs_churn(rows: list[dict[str, str]], output_dir: Path) -> Path:
    not_churned_ages = [float(row["Age"]) for row in rows if row[TARGET_COLUMN] == "0"]
    churned_ages = [float(row["Age"]) for row in rows if row[TARGET_COLUMN] == "1"]

    plt.figure(figsize=(8, 5))
    plt.boxplot([not_churned_ages, churned_ages], labels=["Not Churned", "Churned"])
    plt.title("Age vs Churn")
    plt.xlabel("Customer Status")
    plt.ylabel("Age")
    plt.tight_layout()

    output_path = output_dir / "age_vs_churn.svg"
    plt.savefig(output_path, format="svg")
    plt.close()
    return output_path


def plot_is_active_member_vs_churn(rows: list[dict[str, str]], output_dir: Path) -> Path:
    rates = exited_rate_by_group(rows, "IsActiveMember")

    categories = ["Inactive (0)", "Active (1)"]
    values = [rates["0"][1], rates["1"][1]]

    plt.figure(figsize=(8, 5))
    plt.bar(categories, values)
    plt.title("IsActiveMember vs Churn")
    plt.xlabel("Membership Activity")
    plt.ylabel("Churn Rate")
    plt.tight_layout()

    output_path = output_dir / "is_active_member_vs_churn.svg"
    plt.savefig(output_path, format="svg")
    plt.close()
    return output_path


def generate_all_eda_visuals(
    data_path: Path = DATA_PATH, output_dir: Path = FIGURES_OUTPUT_DIR
) -> list[Path]:
    ensure_output_dir(output_dir)
    rows = load_rows(data_path)

    generated = [
        plot_churn_distribution(rows, output_dir),
        plot_geography_vs_churn(rows, output_dir),
        plot_age_vs_churn(rows, output_dir),
        plot_is_active_member_vs_churn(rows, output_dir),
    ]
    return generated


if __name__ == "__main__":
    paths = generate_all_eda_visuals()
    for path in paths:
        print(f"Saved: {path}")