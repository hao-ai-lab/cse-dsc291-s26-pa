"""Benchmark harness for Part 1.3.

Use this as a starting point. To get full credit on the benchmark you should
sweep at least one of {batch_size, hidden_dim, num_experts, topk} and produce a
table or plot in `analysis.md`.

Run with (from the `pa3/` directory, same convention as `test_moe.py`):
    mpirun --oversubscribe -n 4 python part1/benchmark.py
"""
import time

import numpy as np

from mpi_wrapper import mpi
from rng import get_rng
from moe import SimpleMoE, MoE_EP, MoE_TP


def run_moe(moe_type, batch_size=8, feature_dim=32, hidden_dim=128,
            output_dim=64, num_experts=None, topk=2, n_iters=10):
    if num_experts is None:
        num_experts = mpi.Get_size()

    if mpi.Get_rank() == 0:
        X = get_rng().randn(batch_size, feature_dim)
    else:
        X = None
    X = mpi.bcast(X, root=0)

    model_cls = {"simple": SimpleMoE, "ep": MoE_EP, "tp": MoE_TP}[moe_type]
    moe = model_cls(
        input_dim=feature_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_experts=num_experts,
        topk=topk,
    )

    _ = moe(X)  # warm up
    mpi.Barrier()

    t0 = time.time()
    for _ in range(n_iters):
        outputs = moe(X)
    mpi.Barrier()
    avg_ms = 1000 * (time.time() - t0) / n_iters

    return outputs, avg_ms


def benchmark_moe():
    configs = [
        # (batch, feature, hidden, output, topk)
        (32, 64, 256, 64, 2),
        (64, 64, 256, 64, 2),
        (32, 128, 512, 128, 2),
    ]
    if mpi.Get_rank() == 0:
        print(f"world_size = {mpi.Get_size()}")
        print(f"{'config':40}  {'simple':>10}  {'tp':>10}  {'ep':>10}")

    for cfg in configs:
        batch, feat, hidden, out, topk = cfg
        kwargs = dict(batch_size=batch, feature_dim=feat, hidden_dim=hidden,
                      output_dim=out, topk=topk)
        _, t_simple = run_moe("simple", **kwargs)
        _, t_tp = run_moe("tp", **kwargs)
        _, t_ep = run_moe("ep", **kwargs)
        if mpi.Get_rank() == 0:
            tag = f"b={batch} d={feat} h={hidden} k={topk}"
            print(f"{tag:40}  {t_simple:10.2f}  {t_tp:10.2f}  {t_ep:10.2f}")


if __name__ == "__main__":
    benchmark_moe()
