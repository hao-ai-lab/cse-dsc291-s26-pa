"""Sanity tests for SimpleMoE / MoE_TP / MoE_EP.

Run with:
    mpirun -n 4 python test_moe.py

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


def test_simple_moe():
    rank = mpi.Get_rank()
    register_rng("expert_with_rank", np.random.RandomState(rank + 100))
    result = run_moe("simple", batch_size=10, feature_dim=10, hidden_dim=10,
                     output_dim=10, num_experts=mpi.Get_size(), topk=mpi.Get_size())
    assert result["outputs"].shape == (10, 10)
    if mpi.Get_rank() == 0:
        print("Simple MoE test passed")


def test_ep_moe():
    rank = mpi.Get_rank()
    register_rng("expert_with_rank", np.random.RandomState(rank + 100))
    result = run_moe("ep", batch_size=10, feature_dim=10, hidden_dim=10,
                    output_dim=10, num_experts=mpi.Get_size(), topk=mpi.Get_size())
    assert result["outputs"].shape == (10, 10)
    if mpi.Get_rank() == 0:
        print("Expert Parallel MoE test passed")


def test_tp_moe():
    rank = mpi.Get_rank()
    register_rng("expert_with_rank", np.random.RandomState(rank + 100))
    result = run_moe("tp", batch_size=10, feature_dim=10, hidden_dim=10,
                    output_dim=10, num_experts=mpi.Get_size(), topk=mpi.Get_size())
    assert result["outputs"].shape == (10, 10)
    if mpi.Get_rank() == 0:
        print("Tensor Parallel MoE test passed")


if __name__ == "__main__":
    test_simple_moe()
    test_ep_moe()
    test_tp_moe()
