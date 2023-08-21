import csv
import random

import numpy as np

import nnabla as nn
import nnabla.utils.save
import nnabla.utils.nnp_graph
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.solvers as S

LIST_COLUMN = ["x__{:d}".format(v) for v in range(10)] + [
    "y__{:d}".format(v) for v in range(10)
]
BATCH_SIZE = 10
MODEL_FILENAME = "model_simple.nnp"


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


def load_csvdata(pathname) -> tuple[list, list]:
    list_x = []
    list_y = []
    with open(pathname, "r") as rf:
        csv_r = csv.reader(rf)
        for _r in csv_r:
            if csv_r.line_num != 1:
                r = [float(v) for v in _r]
                list_x.append(r[0:10])
                list_y.append(r[10:20])

    return list_x, list_y


def build(in_x: nn.Variable):
    h = PF.affine(in_x, 50, name="affine_1")
    h = F.relu(h)
    h = PF.affine(h, 10, name="affine_2")
    f = F.relu(h)

    return f


def train(f, in_x, in_y, list_x, list_y, epoch_count: int = 10):
    h = F.squared_error(f, in_y)
    loss = F.mean(h)
    solver = S.Adam()
    solver.set_parameters(nn.get_parameters())

    batch_count = len(list_x) // BATCH_SIZE

    for epoch in range(1, epoch_count + 1):
        print("epoch {:d}".format(epoch))

        for n in range(0, batch_count, BATCH_SIZE):
            x = list_x[n : n + BATCH_SIZE]
            y = list_y[n : n + BATCH_SIZE]

            in_x.d = np.reshape(x, [BATCH_SIZE, 10])
            in_y.d = np.reshape(y, [BATCH_SIZE, 10])
            loss.forward()
            solver.zero_grad()
            loss.backward()
            solver.update()

        print("loss = %.8f" % loss.d)
        print("epoch {:d} done.".format(epoch))


def main():
    random.seed(100)
    generate_dataset("dataset/simple_t.csv", 16384)
    random.seed(200)
    generate_dataset("dataset/simple_v.csv", 256)
    list_train_x, list_train_y = load_csvdata("dataset/simple_t.csv")

    x = nn.Variable(shape=(BATCH_SIZE, 10))
    y = nn.Variable(shape=(BATCH_SIZE, 10))
    f = build(x)
    train(f, x, y, list_train_x, list_train_y, 100)

    contents = {
        "networks": [
            {
                "name": "net1",
                "batch_size": 1,
                "names": {"x0": x},
                "outputs": {"y0": f},
            }
        ],
        "executors": [
            {
                "name": "runtime",
                "network": "net1",
                "data": ["x0"],
                "output": ["y0"],
            }
        ],
    }

    # https://nnabla.readthedocs.io/ja/latest/python/api/utils/save_load.html
    nnabla.utils.save.save(MODEL_FILENAME, contents)


if __name__ == "__main__":
    main()
