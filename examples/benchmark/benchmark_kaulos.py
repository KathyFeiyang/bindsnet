import os
import torch
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from time import time as t
# from experiments import ROOT_DIR
# from bindsnet import ROOT_DIR
ROOT_DIR = ""

from bindsnet.network import Network
from bindsnet.network.topology import Connection
from bindsnet.network.nodes import Input, LIFNodes
from bindsnet.encoding import poisson

# from experiments.benchmark import plot_benchmark

# plots_path = os.path.join(ROOT_DIR, "figures")
benchmark_path = os.path.join(ROOT_DIR, "benchmark")
if not os.path.isdir(benchmark_path):
    os.makedirs(benchmark_path)

# "Warm up" the GPU.
# torch.set_default_tensor_type("torch.cuda.FloatTensor")
# x = torch.rand(1000)
# del x

ms = 1
defaultclock = 1.0 * ms


"""
Benchmark the following two circuit structures and compare with Kaulos:
1. Input(50) => Dense(1024) => LIF(1024)
2. Input(50) => Dense(1024) => Densely-connected-LIF(1024)
"""


def dense_LIF_cpu(n_inputs, n_neurons, time):
    t0 = t()

    torch.set_default_tensor_type("torch.FloatTensor")

    t1 = t()

    network = Network()
    network.add_layer(Input(n=n_inputs), name="input")
    network.add_layer(LIFNodes(n=n_neurons), name="LIF")
    network.add_connection(
        Connection(source=network.layers["input"], target=network.layers["LIF"]),
        source="input",
        target="LIF",
    )

    data = {"X": poisson(datum=torch.rand(n_neurons), time=time)}
    network.run(inputs=data, time=time)

    return t() - t0, t() - t1


def dense_connected_LIF_cpu(n_inputs, n_neurons, time):
    t0 = t()

    torch.set_default_tensor_type("torch.FloatTensor")

    t1 = t()

    network = Network()
    network.add_layer(Input(n=n_inputs), name="input")
    network.add_layer(LIFNodes(n=n_neurons), name="LIF")
    network.add_connection(
        Connection(source=network.layers["input"], target=network.layers["LIF"]),
        source="input",
        target="LIF",
    )
    network.add_connection(
        Connection(source=network.layers["LIF"], target=network.layers["LIF"],
                   w=torch.randint(2, size=[n_neurons] * 2, dtype=torch.float)),
        source="LIF",
        target="LIF",
    )

    data = {"X": poisson(datum=torch.rand(n_neurons), time=time)}
    network.run(inputs=data, time=time)

    return t() - t0, t() - t1


def dense_LIF_gpu(n_inputs, n_neurons, time):
    if torch.cuda.is_available():
        t0 = t()

        torch.set_default_tensor_type("torch.cuda.FloatTensor")

        t1 = t()

        network = Network()
        network.add_layer(Input(n=n_inputs), name="input")
        network.add_layer(LIFNodes(n=n_neurons), name="LIF")
        network.add_connection(
            Connection(source=network.layers["input"], target=network.layers["LIF"]),
            source="input",
            target="LIF",
        )

        data = {"X": poisson(datum=torch.rand(n_neurons), time=time)}
        network.run(inputs=data, time=time)

        return t() - t0, t() - t1


def dense_connected_LIF_gpu(n_inputs, n_neurons, time):
    if torch.cuda.is_available():
        t0 = t()

        torch.set_default_tensor_type("torch.cuda.FloatTensor")

        t1 = t()

        network = Network()
        network.add_layer(Input(n=n_inputs), name="input")
        network.add_layer(LIFNodes(n=n_neurons), name="LIF")
        network.add_connection(
            Connection(source=network.layers["input"], target=network.layers["LIF"]),
            source="input",
            target="LIF",
        )
        network.add_connection(
            Connection(source=network.layers["LIF"], target=network.layers["LIF"],
                    w=torch.randint(2, size=[n_neurons] * 2, dtype=torch.float)),
            source="LIF",
            target="LIF",
        )

        data = {"X": poisson(datum=torch.rand(n_neurons), time=time)}
        network.run(inputs=data, time=time)

        return t() - t0, t() - t1


def main(start=100, stop=1000, step=100, time=1000, interval=100, plot=False):
    f = os.path.join(benchmark_path, f"benchmark_{start}_{stop}_{step}_{time}.csv")
    if os.path.isfile(f):
        os.remove(f)

    n_inputs = 50
    n_neurons = 1024

    times = {
        "dense_LIF_cpu": [],
        "dense_connected_LIF_cpu": [],
        # "dense_LIF_cpu": [],
        # "dense_connected_LIF_gpu": []
    }

    print(f"\nRunning benchmark with {n_neurons} neurons.")
    for framework in times.keys():
        print(f"- {framework}:", end=" ")

        fn = globals()[framework]
        total, sim = fn(n_inputs=n_inputs, n_neurons=n_neurons, time=time)
        times[framework].append(sim)

        print(f"(total: {total:.4f}; sim: {sim:.4f})")

    print(times)

    df = pd.DataFrame.from_dict(times)
    df.index = [n_neurons]

    print()
    print(df)
    print()

    df.to_csv(f)

    # plot_benchmark.main(
    #     start=start, stop=stop, step=step, time=time, interval=interval, plot=plot
    # )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=100)
    parser.add_argument("--stop", type=int, default=1000)
    parser.add_argument("--step", type=int, default=100)
    parser.add_argument("--time", type=int, default=1000)
    parser.add_argument("--interval", type=int, default=100)
    parser.add_argument("--plot", dest="plot", action="store_true")
    parser.set_defaults(plot=False)
    args = parser.parse_args()

    main(
        start=args.start,
        stop=args.stop,
        step=args.step,
        time=args.time,
        interval=args.interval,
        plot=args.plot,
    )
