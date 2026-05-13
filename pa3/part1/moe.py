"""Mixture-of-Experts: reference, tensor-parallel, and expert-parallel variants.

You will implement `ShardedLinear`, `MoE_TP`, and `MoE_EP` in this file. The
reference `SimpleMoE` and a pre-built `Router` are provided.
"""
import numpy as np

from mpi_wrapper import mpi
from rng import get_rng, rng_context


class Linear:
    """Simple linear layer y = xW + b."""

    def __init__(self, in_features, out_features):
        self.weight = get_rng().randn(in_features, out_features) * 0.01
        self.bias = np.zeros(out_features)

    def __call__(self, x):
        return np.dot(x, self.weight) + self.bias


class Expert:
    """Two-layer MLP expert with ReLU."""

    def __init__(self, input_dim, hidden_dim, output_dim):
        with rng_context("expert"):
            self.fc1 = Linear(input_dim, hidden_dim)
            self.fc2 = Linear(hidden_dim, output_dim)

    def __call__(self, x):
        hidden = self.fc1(x)
        hidden = np.maximum(0, hidden)  # ReLU
        return self.fc2(hidden)


class Router:
    """Softmax-gated top-k router (replicated across ranks)."""

    def __init__(self, input_dim, num_experts):
        self.linear = Linear(input_dim, num_experts)

    def __call__(self, x, topk=1):
        logits = self.linear(x)
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        indices = np.argsort(-probs, axis=1)[:, :topk]
        gates = np.take_along_axis(probs, indices, axis=1)
        gates = gates / np.sum(gates, axis=1, keepdims=True)
        return indices, gates


# ---------------------------------------------------------------------------
# Reference implementation: not parallel. Use this to verify correctness.
# ---------------------------------------------------------------------------
class SimpleMoE:
    def __init__(self, input_dim, hidden_dim, output_dim, num_experts, topk=1):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_experts = num_experts
        self.topk = min(topk, num_experts)

        with rng_context("router"):
            self.router = Router(input_dim, num_experts)

        with rng_context("expert"):
            self.experts = [
                Expert(input_dim, hidden_dim, output_dim) for _ in range(num_experts)
            ]

    def forward(self, x):
        batch_size = x.shape[0]
        indices, gates = self.router(x, self.topk)
        outputs = np.zeros((batch_size, self.output_dim))
        for k in range(self.topk):
            for i in range(batch_size):
                expert_idx = indices[i, k]
                gate = gates[i, k]
                item = x[i : i + 1]
                expert_output = self.experts[expert_idx](item)
                outputs[i] += gate * expert_output[0]
        return outputs

    def __call__(self, x):
        return self.forward(x)


# ---------------------------------------------------------------------------
# Part 1.1 — Tensor Parallel MoE.
# ---------------------------------------------------------------------------
class ShardedLinear:
    """Linear layer whose weight is column-sharded across MPI ranks.

    Each rank stores a `(in_features, out_features // world_size)` slice of the
    weight matrix. The forward pass produces the *full* output of shape
    `(batch, out_features)` on every rank, which means a collective is required
    to reassemble the columns each rank computed.

    Requires that `out_features` is evenly divisible by the world size.
    """

    def __init__(self, in_features, out_features):
        self.rank = mpi.Get_rank()
        self.world_size = mpi.Get_size()

        assert out_features % self.world_size == 0, (
            f"Output features ({out_features}) must be evenly divisible by "
            f"world size ({self.world_size})"
        )

        self.in_features = in_features
        self.out_features_global = out_features
        self.local_out_features = out_features // self.world_size
        self.output_offset = self.rank * self.local_out_features

        # Each rank initializes only its local slice. Because the rng has the
        # same seed on every rank inside the `expert` context, you would get
        # *identical* slices on each rank if you naively called randn here, so
        # callers should wrap construction inside `rng_context("expert")` and
        # advance the rng deterministically per rank if they want distinct
        # initial weights. For grading we only check the forward semantics.
        self.weight = get_rng().randn(in_features, self.local_out_features) * 0.01
        self.bias = get_rng().randn(self.local_out_features)

    def __call__(self, x):
        if x.shape[0] == 0:
            return np.zeros((0, self.out_features_global), dtype=np.float32)

        result = np.zeros((x.shape[0], self.out_features_global), dtype=np.float32)

        # TODO (Part 1.1): produce the full output of the sharded linear layer.
        #   - Compute the local partial output  x @ self.weight + self.bias
        #     into the slice of `result` that this rank owns.
        #   - Use a collective to make every rank hold the *full* output.
        #     `Allreduce` (sum) on the zero-padded result is one valid choice.
        #     `Allgather` along the column dimension is another.
        return result


class ShardedExpert:
    """Expert whose weights are sharded along the hidden / output dim."""

    def __init__(self, input_dim, hidden_dim, output_dim):
        with rng_context("expert"):
            self.fc1 = ShardedLinear(input_dim, hidden_dim)
            self.fc2 = ShardedLinear(hidden_dim, output_dim)

    def __call__(self, x):
        hidden = self.fc1(x)
        hidden = np.maximum(0, hidden)
        return self.fc2(hidden)


class MoE_TP:
    """Mixture-of-Experts with tensor-parallel experts.

    Every rank holds a slice of every expert. Routing is replicated. After
    each expert's forward pass, ranks need a collective to reassemble the
    full output of that expert before applying the gate.
    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_experts, topk=1):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_experts = num_experts
        self.topk = min(topk, num_experts)
        self.rank = mpi.Get_rank()
        self.world_size = mpi.Get_size()

        with rng_context("router"):
            self.router = Router(input_dim, num_experts)

        with rng_context("expert"):
            self.experts = [
                ShardedExpert(input_dim, hidden_dim, output_dim)
                for _ in range(num_experts)
            ]

        if self.rank == 0:
            print(
                f"[MoE_TP] world_size={self.world_size}, num_experts={num_experts}, topk={self.topk}"
            )

    def forward(self, x):
        """
        Args:
            x: `(batch_size, input_dim)` — replicated on every rank.

        Returns:
            `(batch_size, output_dim)` — replicated on every rank.
        """
        batch_size = x.shape[0]
        outputs = np.zeros((batch_size, self.output_dim))

        # TODO (Part 1.1): implement the TP-style forward pass.
        # 1. Use `self.router(x, self.topk)` to get routing indices and gates.
        # 2. For each (token, top-k slot), feed the token through the assigned
        #    expert. Each `ShardedExpert` already handles the cross-rank
        #    collective internally, so you can call it like a regular Expert.
        # 3. Combine the gated expert outputs into `outputs`.
        indices, gates = self.router(x, self.topk)

        return outputs

    def __call__(self, x):
        return self.forward(x)


# ---------------------------------------------------------------------------
# Part 1.2 — Expert Parallel MoE.
# ---------------------------------------------------------------------------
class MoE_EP:
    """Mixture-of-Experts with expert-parallel experts.

    Each rank owns *exactly one* expert. After routing, tokens that have been
    assigned to expert `e` must be sent to the rank that owns expert `e`. The
    expert computes its forward pass on the tokens it received and the results
    are sent back to the originating ranks.

    The natural collective for this pattern is **all-to-all**: each rank
    builds `world_size` buckets (one per destination rank) and exchanges them.
    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_experts, topk=1):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_experts = num_experts  # == world size
        self.topk = min(topk, self.num_experts)
        self.rank = mpi.Get_rank()
        self.world_size = mpi.Get_size()

        assert num_experts == self.world_size, (
            "MoE_EP assumes one expert per rank; got "
            f"num_experts={num_experts}, world_size={self.world_size}"
        )

        with rng_context("router"):
            self.router = Router(input_dim, self.num_experts)

        # Each rank initializes its own expert independently — we want the
        # experts to be different, so this rng is rank-specific.
        with rng_context("expert_with_rank"):
            self.expert = Expert(input_dim, hidden_dim, output_dim)

    def forward(self, x):
        """
        Args:
            x: `(batch_size, input_dim)` — replicated on every rank.

        Returns:
            `(batch_size, output_dim)` — replicated on every rank.
        """
        batch_size = x.shape[0]
        outputs = np.zeros((batch_size, self.output_dim))

        # TODO (Part 1.2): implement the EP-style forward pass.
        # 1. Use `self.router(x, self.topk)` to get routing indices and gates.
        # 2. For each top-k slot, build a list of length `world_size` where
        #    bucket[r] is the dict / array of tokens that should be processed
        #    by rank r's expert. Track the original token indices so you can
        #    scatter results back into `outputs`.
        # 3. Use `mpi.alltoall(buckets)` to exchange tokens. Each rank now has
        #    the tokens it needs to process locally.
        # 4. Run `self.expert(local_tokens)` on the received tokens.
        # 5. All-to-all the results back to the originating ranks.
        # 6. Apply gates and accumulate into `outputs`.
        #
        # Note: `mpi.alltoall` (lower-case) accepts arbitrary Python objects,
        # so you can send NumPy arrays of differing sizes per destination.
        indices, gates = self.router(x, self.topk)

        return outputs

    def __call__(self, x):
        return self.forward(x)
