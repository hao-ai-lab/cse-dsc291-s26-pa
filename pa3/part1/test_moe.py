"""Sanity tests for SimpleMoE / MoE_TP / MoE_EP.

Run from the pa3/ root (use --oversubscribe if you have < 4 cores):
    mpirun -n 4 python part1/test_moe.py

For MoE_EP we require num_experts == world_size.
"""
import time

import numpy as np

from mpi_wrapper import mpi
from rng import get_rng, register_rng
from moe import SimpleMoE, MoE_EP, MoE_TP


def run_moe(
    moe_type="tp",
    batch_size=8,
    feature_dim=32,
    hidden_dim=128,
    output_dim=64,
    num_experts=None,
    topk=2,
):
    """Run one of the three MoE variants and time the forward pass."""
    if num_experts is None:
        num_experts = mpi.Get_size()

    np.random.seed(0)
    X = np.random.randn(batch_size, feature_dim)

    if moe_type != "simple":
        # Make sure every rank sees the same input.
        if mpi.Get_rank() == 0:
            X = get_rng().randn(batch_size, feature_dim)
        else:
            X = None
        X = mpi.bcast(X, root=0)

    model_cls = {"simple": SimpleMoE, "ep": MoE_EP, "tp": MoE_TP}.get(moe_type, MoE_TP)
    moe = model_cls(
        input_dim=feature_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_experts=num_experts,
        topk=topk,
    )

    _ = moe(X)  # warm up

    N = 3
    t0 = time.time()
    for _ in range(N):
        outputs = moe(X)
    avg_ms = 1000 * (time.time() - t0) / N

    if mpi.Get_rank() == 0:
        print(f"[{moe_type:>6}] forward avg = {avg_ms:7.2f} ms")

    return dict(outputs=outputs, avg_duration_ms=avg_ms)


# ShardedLinear requires hidden_dim / output_dim to be divisible by the world
# size, so derive the dimensions from the world size to keep `mpirun -n <N>`
# working for any N.
def _dims():
    ws = mpi.Get_size()
    return dict(batch_size=8, feature_dim=4 * ws, hidden_dim=8 * ws,
                output_dim=4 * ws)


def test_simple_moe():
    rank = mpi.Get_rank()
    register_rng("expert_with_rank", np.random.RandomState(rank + 100))
    d = _dims()
    result = run_moe("simple", num_experts=mpi.Get_size(), topk=2, **d)
    assert result["outputs"].shape == (d["batch_size"], d["output_dim"])
    if mpi.Get_rank() == 0:
        print("Simple MoE test passed")


def test_ep_moe():
    rank = mpi.Get_rank()
    register_rng("expert_with_rank", np.random.RandomState(rank + 100))
    d = _dims()
    result = run_moe("ep", num_experts=mpi.Get_size(), topk=2, **d)
    assert result["outputs"].shape == (d["batch_size"], d["output_dim"])
    if mpi.Get_rank() == 0:
        print("Expert Parallel MoE test passed")


def test_tp_moe():
    rank = mpi.Get_rank()
    register_rng("expert_with_rank", np.random.RandomState(rank + 100))
    d = _dims()
    result = run_moe("tp", num_experts=mpi.Get_size(), topk=2, **d)
    assert result["outputs"].shape == (d["batch_size"], d["output_dim"])
    if mpi.Get_rank() == 0:
        print("Tensor Parallel MoE test passed")


if __name__ == "__main__":
    test_simple_moe()
    test_ep_moe()
    test_tp_moe()
