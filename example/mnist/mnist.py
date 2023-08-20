import os
import gzip
import array
import struct

import numpy as np
import httpx
import PIL.Image

import nnabla as nn
import nnabla.utils.save
import nnabla.utils.nnp_graph
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.solvers as S

MNIST_URL = "https://azureopendatastorage.blob.core.windows.net/mnist"
MNIST_FILES = [
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
]

BATCH_SIZE = 100


def download_data(url, filename):
    if os.path.exists(filename) is False:
        with open(filename, "wb") as wf:
            res = httpx.get("{:s}/{:s}".format(url, filename))
            wf.write(res.content)


def convert_data(filename, label: bool):
    with gzip.open(filename) as rf:
        struct.unpack("I", rf.read(4))
        item_size = struct.unpack(">I", rf.read(4))[0]
        if label is True:
            list_result = []
            for v in array.array("B", rf.read()):
                x = [0.0] * 10
                x[v] = 1.0
                list_result.append(x)
            return list_result
        else:
            w = struct.unpack(">I", rf.read(4))[0]
            h = struct.unpack(">I", rf.read(4))[0]

            res = np.frombuffer(rf.read(item_size * w * h), dtype=np.uint8)
            res = res.reshape(item_size, w * h)
            return res / 255.0


def build(in_x, in_y):
    with nn.parameter_scope("affine1"):
        h = PF.affine(in_x, 50, name="affine1")
        h = F.relu(h)
    with nn.parameter_scope("affine2"):
        h = PF.affine(h, 20, name="affine2")
        h = F.relu(h)
    with nn.parameter_scope("affine3"):
        h = PF.affine(h, 10, name="affine3")
        f = F.relu(h)

    contents = {
        "networks": [
            {
                "name": "net1",
                "batch_size": 1,
                "outputs": {"y0": f},
                "names": {"x0": in_x},
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
    return f, contents


def train(f, in_x, in_y, dict_mnist: dict, epoch_count: int = 10):
    h = F.squared_error(f, in_y)
    loss = F.mean(h)
    solver = S.Adam()
    solver.set_parameters(nn.get_parameters())

    train_d = dict_mnist["train-images-idx3-ubyte.gz"]
    train_l = dict_mnist["train-labels-idx1-ubyte.gz"]
    data_size = len(train_d)

    for epoch in range(1, epoch_count + 1):
        print("epoch {:d}".format(epoch))

        for n in range(data_size // BATCH_SIZE):
            vx = train_d[n : n + BATCH_SIZE]
            vy = np.array(train_l[n : n + BATCH_SIZE])

            in_x.d = np.reshape(vx, [BATCH_SIZE, 784])
            in_y.d = np.reshape(vy, [BATCH_SIZE, 10])
            loss.forward()
            solver.zero_grad()
            loss.backward()
            solver.update()
            if ((n + 1) % 100) == 0:
                print("loss = %.8f" % loss.d)

        print("epoch {:d} done.".format(epoch))


def inference(dict_mnist: dict, modelfile="model_mnist.nnp"):
    nnp = nnabla.utils.nnp_graph.NnpLoader(modelfile)
    net = nnp.get_network("net1", batch_size=1)

    in_x = net.inputs["x0"]
    f = net.outputs["y0"]

    test_d = dict_mnist["t10k-images-idx3-ubyte.gz"]
    test_l = dict_mnist["t10k-labels-idx1-ubyte.gz"]
    data_size = len(test_d)

    valid_count = 0
    for n in range(data_size):
        vx = test_d[n]
        vy = test_l[n]

        in_x.d = np.reshape(vx, [1, 784])
        f.forward()

        list_v = list(f.d[0])
        pos = list_v.index(max(list_v))
        if test_l[n][pos] == 1.0:
            valid_count += 1

    print("{:3.2f}".format((valid_count / data_size) * 100))


def main():
    dict_data = {}
    for filename in MNIST_FILES:
        download_data(MNIST_URL, filename)
        dict_data[filename] = convert_data(filename, filename.find("label") > -1)
        print(filename)

    if True:
        x = nn.Variable(shape=(BATCH_SIZE, 784))
        y = nn.Variable(shape=(BATCH_SIZE, 10))
        f, contents = build(x, y)
        train(f, x, y, dict_data, 10)
        nnabla.utils.save.save("model_mnist.nnp", contents)

    inference(dict_data)


if __name__ == "__main__":
    main()
