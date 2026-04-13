from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.config import DATA_PATH, FIGURES_OUTPUT_DIR, TARGET_COLUMN
from src.data_loader import load_rows
from src.eda_analysis import exited_rate_by_group


def ensure_output_dir(output_dir: Path) -> None:
    # Create the output directory if it does not exist yet.
    # `parents=True` allows Python to create any missing parent folders, and
    # `exist_ok=True` avoids raising an error if the folder already exists.
    output_dir.mkdir(parents=True, exist_ok=True)


def plot_churn_distribution(rows: list[dict[str, str]], output_dir: Path) -> Path:
    # Count how many rows belong to each churn class.
    counts = {"0": 0, "1": 0}
    for row in rows:
        counts[row[TARGET_COLUMN]] += 1

    # Build the bar labels and matching values in the order we want to display.
    labels = ["Retained", "Churned"]
    values = [counts["0"], counts["1"]]
    colors = ["green", "red"]
    total_customers = len(rows)

    # Create a new figure and draw a simple bar chart.
    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, values, color=colors)
    plt.title("Churn Distribution")
    plt.xlabel("Customer Status")
    plt.ylabel("Customer Count")
    # Add extra headroom above the tallest bar so the text labels do not touch
    # the top edge of the chart. The requested upper bound of 10,000 is kept,
    # but a little extra space is added beyond it.
    plt.ylim(0, 10500)

    # Add the total customer count and percentage above each bar.
    for bar, value in zip(bars, values):
        percentage = value / total_customers
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + 120,
            f"{value:,} ({percentage:.1%})",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    # Adjust spacing so labels and titles do not overlap.
    plt.tight_layout()

    # Save the chart as an SVG file and then close the figure so memory is freed.
    output_path = output_dir / "churn_distribution.svg"
    plt.savefig(output_path, format="svg")
    plt.close()
    return output_path


def plot_geography_vs_churn(rows: list[dict[str, str]], output_dir: Path) -> Path:
    # Build a dictionary that tracks churned and non-churned customer counts
    # for each geography.
    geography_counts: dict[str, dict[str, int]] = {}
    for row in rows:
        geography = row["Geography"]
        exited = row[TARGET_COLUMN]

        # Initialize the geography bucket the first time it is seen.
        if geography not in geography_counts:
            geography_counts[geography] = {"0": 0, "1": 0}
        # Increment the correct churn-status count for this geography.
        geography_counts[geography][exited] += 1

    # Split the dictionary into ordered lists for plotting.
    geographies = list(geography_counts.keys())
    retained = [geography_counts[g]["0"] for g in geographies]
    churned = [geography_counts[g]["1"] for g in geographies]
    churn_rates = [
        geography_counts[g]["1"] / (geography_counts[g]["0"] + geography_counts[g]["1"])
        for g in geographies
    ]

    # Prepare x-axis positions and a shared width for grouped bars.
    x = np.arange(len(geographies))
    width = 0.35

    # Create a grouped bar chart with one retained bar and one churned bar per country.
    plt.figure(figsize=(8, 5))
    retained_bars = plt.bar(x - width / 2, retained, width=width, color="green", label="Retained")
    churned_bars = plt.bar(x + width / 2, churned, width=width, color="red", label="Churned")
    plt.xticks(list(x), geographies)
    plt.title("Geography vs Churn")
    # Increase label padding so the x-axis label appears below the custom
    # churn-rate annotations that sit under each country name.
    plt.xlabel("Geography", labelpad=38)
    plt.ylabel("Customer Count")
    plt.legend()
    # Leave enough headroom above the tallest grouped bar for the count labels.
    plt.ylim(0, max(retained + churned) * 1.2)

    grand_total = len(rows)

    # Add the raw count and percentage above each retained bar.
    for bar, value in zip(retained_bars, retained):
        percentage = value / grand_total
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + 25,
            f"{value:,} ({percentage:.1%})",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # Add the raw count and percentage above each churned bar.
    for bar, value in zip(churned_bars, churned):
        percentage = value / grand_total
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + 25,
            f"{value:,} ({percentage:.1%})",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # Add each country's churn rate below the country label area.
    for x_pos, churn_rate in zip(x, churn_rates):
        plt.text(
            x_pos,
            -0.14,
            f"churn rate: {churn_rate:.1%}",
            ha="center",
            va="top",
            fontsize=9,
            transform=plt.gca().get_xaxis_transform(),
        )

    plt.tight_layout()

    # Save the finished chart and close the current figure.
    output_path = output_dir / "geography_churned.svg"
    plt.savefig(output_path, format="svg")
    plt.close()
    return output_path


def plot_age_vs_churn(rows: list[dict[str, str]], output_dir: Path) -> Path:
    # Group ages into 10-year ranges so the chart shows how churn rate changes
    # across age bands instead of individual raw ages.
    age_buckets: dict[str, list[int]] = {}
    for row in rows:
        age = int(float(row["Age"]))
        bucket_start = (age // 10) * 10
        bucket_label = f"{bucket_start}-{bucket_start + 9}"

        if bucket_label not in age_buckets:
            age_buckets[bucket_label] = [0, 0]

        # Position 0 stores the total number of customers in the bucket.
        age_buckets[bucket_label][0] += 1
        # Position 1 stores how many of those customers churned.
        age_buckets[bucket_label][1] += int(row[TARGET_COLUMN])

    # Sort the age buckets numerically so they appear left-to-right in age order.
    sorted_labels = sorted(age_buckets.keys(), key=lambda label: int(label.split("-")[0]))
    churn_rates = [
        age_buckets[label][1] / age_buckets[label][0]
        for label in sorted_labels
    ]

    # Draw a bar chart with age ranges on the x-axis and churn rate on the y-axis.
    plt.figure(figsize=(8, 5))
    bars = plt.bar(sorted_labels, churn_rates, color="steelblue")
    plt.title("Age vs Churn")
    plt.xlabel("Age Range")
    plt.ylabel("Churn Rate")
    plt.xticks(rotation=45)
    # Increase the y-axis ceiling so the percentage labels fit above the bars.
    plt.ylim(0, max(churn_rates) * 1.2 if churn_rates else 1)

    # Add the churn-rate percentage above each age-range bar.
    for bar, rate in zip(bars, churn_rates):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            rate + 0.005,
            f"{rate:.1%}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()

    # Save the age-range churn-rate chart and close the figure.
    output_path = output_dir / "age_vs_churn.svg"
    plt.savefig(output_path, format="svg")
    plt.close()
    return output_path


def plot_is_active_member_vs_churn(rows: list[dict[str, str]], output_dir: Path) -> Path:
    # Reuse the churn-rate helper from the analysis module so the churn rate is
    # computed consistently in both the text report and the figures.
    rates = exited_rate_by_group(rows, "IsActiveMember")

    # Build readable category labels and matching churn-rate values.
    categories = ["Inactive (0)", "Active (1)"]
    values = [rates["0"][1], rates["1"][1]]
    totals = [rates["0"][0], rates["1"][0]]

    # Plot churn rate by active membership status.
    plt.figure(figsize=(8, 5))
    bars = plt.bar(categories, values, color=["orange", "purple"])
    plt.title("IsActiveMember vs Churn")
    plt.xlabel("Membership Activity")
    plt.ylabel("Churn Rate")
    # Leave headroom for the rate and sample-size labels above each bar.
    plt.ylim(0, max(values) * 1.25 if values else 1)

    # Show the churn rate and total customers represented by each bar.
    for bar, rate, total in zip(bars, values, totals):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            rate + 0.005,
            f"{rate:.1%}\n(n={total:,})",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()

    # Save the chart and close the figure.
    output_path = output_dir / "is_active_member_vs_churn.svg"
    plt.savefig(output_path, format="svg")
    plt.close()
    return output_path


def generate_all_eda_visuals(
    data_path: Path = DATA_PATH, output_dir: Path = FIGURES_OUTPUT_DIR
) -> list[Path]:
    # Make sure the destination folder exists before saving any figures.
    ensure_output_dir(output_dir)
    # Load the raw dataset once and reuse the rows for all plots.
    rows = load_rows(data_path)

    # Generate each chart and keep the resulting file paths in a list.
    generated = [
        plot_churn_distribution(rows, output_dir),
        plot_geography_vs_churn(rows, output_dir),
        plot_age_vs_churn(rows, output_dir),
        plot_is_active_member_vs_churn(rows, output_dir),
    ]
    return generated


if __name__ == "__main__":
    # Allow this module to be run directly so visuals can be generated without
    # going through the main project entry point.
    paths = generate_all_eda_visuals()
    for path in paths:
        print(f"Saved: {path}")
