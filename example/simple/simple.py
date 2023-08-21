import csv
import random

LIST_COLUMN = ["x__{:d}".format(v) for v in range(10)] + [
    "y__{:d}".format(v) for v in range(10)
]


def generate_dataset(pathname: str, rows: int):
    with open(pathname, "w", newline="") as wf:
        csv_w = csv.writer(wf)
        csv_w.writerow(LIST_COLUMN)

        for n in range(rows):
            list_record_x = [random.random() for v in range(10)]
            idx = list_record_x.index(max(list_record_x))
            list_record_y = [0.0] * 10
            list_record_y[idx] = 1.0

            csv_w.writerow(list_record_x + list_record_y)


def main():
    generate_dataset("dataset/simple_t.csv", 16384)
    generate_dataset("dataset/simple_v.csv", 256)


if __name__ == "__main__":
    main()
