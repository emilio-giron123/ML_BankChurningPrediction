import csv
from pathlib import Path


def load_rows(path: Path) -> list[dict[str, str]]:
    # Open the CSV file using UTF-8 with BOM support so the code still works
    # if the file contains a byte order mark at the beginning.
    with path.open(newline="", encoding="utf-8-sig") as csv_file:
        # `csv.DictReader` reads the header row first and then turns every
        # remaining row into a dictionary:
        # - keys are column names from the header
        # - values are cell contents stored as strings
        #
        # Wrapping it with `list(...)` loads the full file into memory so the
        # rest of the project can iterate over the rows multiple times.
        return list(csv.DictReader(csv_file))
