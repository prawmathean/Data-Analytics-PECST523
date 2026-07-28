import csv
import random

NUM_MIN_RANGE = 0
NUM_MAX_RANGE = 1000
TOTAL_COUNT = 10000
WRONG_CHANCE_PERCENT = 5  # % of rows that get a bogus "sum"

correct_count = 0
wrong_count = 0

with open("train_data.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["X", "Y", "sum"])  # header row so pandas can read column names

    for _ in range(TOTAL_COUNT):
        x = random.randint(NUM_MIN_RANGE, NUM_MAX_RANGE)
        y = random.randint(NUM_MIN_RANGE, NUM_MAX_RANGE)

        if random.randint(0, 100) <= (100 - WRONG_CHANCE_PERCENT):
            writer.writerow([x, y, x + y])
            correct_count += 1
        else:
            # deliberately wrong label, to simulate noisy data
            writer.writerow([x, y, random.randint(1, NUM_MAX_RANGE)])
            wrong_count += 1

print(f"Total entries = {TOTAL_COUNT}, wrong entries = {wrong_count}, correct entries = {correct_count}")
print(f"% of wrong entries = {(wrong_count / TOTAL_COUNT) * 100:.2f}%")
