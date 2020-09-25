# solution
import csv
from pathlib import Path

names = []

file_path = Path.home() / "favorite_colors.csv"

with file_path.open(mode="r", encoding="utf-8") as file:
    reader = csv.DictReader(file)
    for row in reader:
        row['name'] = "ivan"
        names.append(row)

for item in names:
    print(item['name'])