from src.eda_analysis import run_analysis_report
from src.eda_visuals import generate_all_eda_visuals


def main() -> None:
    run_analysis_report()
    paths = generate_all_eda_visuals()
    for path in paths:
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()