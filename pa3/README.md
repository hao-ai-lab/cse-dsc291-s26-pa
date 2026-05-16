# CSE 291 / DSC 291 — Programming Assignment 3

Welcome to PA3! Start early. See the course site / Gradescope for the deadline
and the late-day policy.

Academic integrity is key. You may discuss ideas with classmates, but do not
copy solutions.

## Collaboration

- Parts 1, 2, and 3 may be done in groups of up to **3** students. Submit one
  tarball per group on Gradescope.
- Part 4 (the essay) must be done **individually**.

## Local self-check

From the `pa3/` directory:

```bash
python local_check.py
```

This is a small sanity tool, **not your grade**. It auto-checks only the
items that are 100% deterministic and unambiguous:

- **Part 1**: your `moe.py` runs under `mpirun` and produces the correct
  output shape, replicated across ranks, and non-trivial (not all-zeros).
- **Part 2**: `model_training_cost_analysis_llama` returns the exact
  Llama-3 8B parameter count, and `get_optimal_N_D_from_cost(5_000_000)`
  selects the correct GPU.

It also **prints** your Part 2 FLOPs / peak memory / (N, D) / DeepSeek-V3
numbers so you can sanity-check them yourself, but does not score them.
Everything else — those Part 2 numbers, `moe.md`, the entire Part 3
notebook + report, and the Part 4 essay — is graded by the course staff
from your submitted files and saved notebook output.

> A green `local_check` does **not** mean full marks, and an item it does
> not check is **not** worth zero — it just means a human grades it. Run
> the Part 3 notebook yourself, save its output, and write the
> analysis/report; that is what the staff read. Do not hardcode the
> printed expected values: the staff re-run the Part 2 functions on
> held-out inputs (a different config, a different budget).

## Submission

From the `pa3/` root run:

```bash
make handin.tar
```

This creates `handin.tar` containing `part1/`, `part2/`, and `part3/`. Upload
it to Gradescope under **PA3**. Submit the Part 4 essay separately as a PDF
under **PA3 — Essay**.

## Environment

```bash
conda create -n cse291pa3 python=3.10 -y
conda activate cse291pa3
pip install -r requirements.txt
```

You will need an MPI implementation installed (Part 1) and a CUDA-capable GPU
or access to one (Part 3 is workable on CPU but the speedup is meaningless).

---

## Part 1: Mixture of Experts (50 pts)

You will implement two distributed Mixture-of-Experts variants — **tensor
parallel (TP)** and **expert parallel (EP)** — and benchmark them against a
serial reference. This part builds directly on the MPI primitives you wrote in
PA2 §2.1: `Allreduce`, `Allgather`, `Alltoall`, and (optionally) your own
`myAllreduce` / `myAlltoall`.

A reference `SimpleMoE` and a working `Router` are provided in `part1/moe.py`.
Skeletons for `ShardedLinear`, `MoE_TP`, and `MoE_EP` are provided.

Test (run from the `pa3/` directory):

```bash
mpirun --oversubscribe -n 4 python part1/test_moe.py
```

> `--oversubscribe` is included because many environments (containers,
> Slurm/cgroup-limited shells) report only one allocatable slot regardless
> of physical core count, and Open MPI otherwise refuses to launch 4 ranks.
> It is harmless on a normal multi-core machine.

### 1.1 Tensor Parallel (20 pts)

Every rank holds a slice (column shard) of every expert. Each rank computes a
partial expert output for the whole batch; ranks then collectively assemble
the full output.

- `ShardedLinear` (10 pts): a column-sharded linear layer that returns the
  full output on every rank. Use either `Allreduce` over a zero-padded local
  output or `Allgather` along the column dim.
- `MoE_TP` (10 pts): full TP forward pass that uses `ShardedExpert` and the
  replicated router.

Grading checks forward semantics — correct output shape and the full
result replicated on every rank — not numerical equality with
`SimpleMoE`. The given per-rank `ShardedLinear` init does not tile
`SimpleMoE`'s weight matrix, so you do **not** need to modify
`ShardedLinear.__init__`; implement only the `__call__` / `forward`
methods.

### 1.2 Expert Parallel (20 pts)

Each rank holds **one** expert in its entirety. After routing, tokens have to
be shipped to the rank that owns the expert they were assigned to, then
results have to be shipped back. This is the canonical use case for
all-to-all.

- `MoE_EP` (20 pts): EP forward pass, using `mpi.alltoall(...)` (the
  pickle-based collective; supports variable-sized buckets per destination).

> **Bonus path (+5 pts, best-effort).** If you copy `myAlltoall` from your
> PA2 `mpi_wrapper/comm.py` into the marked location in
> `part1/mpi_wrapper/comm.py` and route the EP all-to-all through it
> (zero-padded so per-rank segment sizes match), you can earn up to +5.
> This is graded by manual inspection of your code and reasoning; there is
> no autograder check for it.

### 1.3 Benchmark (10 pts)

- Modify `part1/benchmark.py` to sweep at least one of {batch size, hidden
  dim, num experts, topk} (5 pts).
- Write a short discussion in `part1/analysis.md` comparing TP vs. EP at
  small / medium / large workloads. Identify whether each variant is
  compute-bound or communication-bound and explain why (5 pts).

---

## Part 2: Scaling Laws and Training Cost Analysis (30 pts + 25 pts bonus)

You will estimate parameter counts, training FLOPs, and peak training memory
for two real models, and design your own model under a fixed compute budget.

### 2.1 Llama-3 8B Cost Analysis (15 pts)

`part2/llama3_8b_config.json` contains the architecture config of
**Llama-3 8B**. Implement `model_training_cost_analysis_llama` in
`part2/model_training_cost_analysis.py`:

- **Total trainable parameters**, including:
  - token embedding (Llama-3 ties no longer hold across all variants — check
    `tie_word_embeddings`),
  - the attention block (Q/K/V/O projections — note the **GQA** config:
    `num_key_value_heads = 8` vs. `num_attention_heads = 32`),
  - the MLP block (gate/up/down with SwiGLU),
  - the RMSNorm layers.
- **Forward FLOPs** of a single transformer layer in TFLOPs. Count the
  attention projections (Q/K/V/O), the **QK^T and attention·V matmuls**,
  and the MLP, with one multiply-add = 2 FLOPs.
  **Use `sequence_length = config["max_position_embeddings"]` and
  `batch_size = 1`** as the grading convention.
- **Peak forward memory** for a single transformer layer under bf16 with
  rematerialization at layer boundaries.

We grade with:

```bash
python part2/model_training_cost_analysis.py --model_config part2/llama3_8b_config.json
```

### 2.2 Design a Model Under a Compute Budget (15 pts)

Use the scaling law

$$L(N, D) = \frac{406.4}{N^{0.34}} + \frac{410.7}{D^{0.29}} + 1.69$$

and a budget of **\$5,000,000**. Pick a GPU among the following options
(assume MFU = 40% across all three):

| GPU  | $/h spot (assumed) | Peak FP16 / BF16 |
|------|-------------------:|-----------------:|
| H100 |             \$3.0  |       989 TFLOPs |
| H200 |             \$4.0  |       989 TFLOPs |
| B200 |             \$6.0  |      2250 TFLOPs |

Implement `get_optimal_N_D_from_cost` to (1) compute the effective FLOPs each
GPU buys for the budget, (2) pick the GPU that maximizes effective FLOPs, and
(3) solve for the optimal `(N, D)` under `6 N D ≈ F_total`.

We grade with:

```bash
python part2/model_training_cost_analysis.py --training_budget 5000000
```

Then create `part2/my_model_config.json` matching the Llama-3 config schema
with hyperparameters that hit your optimal `N`. This file is reviewed by
hand — you do not need to (and should not) run it through the CLI, which
only recognizes config filenames containing `llama` or `deepseek`.

### 2.3 MoE Cost Analysis — DeepSeek-V3 (Bonus, 25 pts)

`part2/deepseek_v3_config.json` contains DeepSeek-V3's config. Implement
`model_training_cost_analysis_deepseek` and report total vs. activated
parameters per token. Mind:

- **MLA** attention (`q_lora_rank`, `kv_lora_rank`, `qk_nope_head_dim`,
  `qk_rope_head_dim`, `v_head_dim`).
- The first `first_k_dense_replace` layers are dense; the rest are MoE with
  `n_routed_experts`, `n_shared_experts`, `num_experts_per_tok`, and
  `moe_intermediate_size`.

Write `part2/moe.md` arguing one concrete advantage and one concrete
disadvantage of MoE relative to a same-budget dense model.

We grade with:

```bash
python part2/model_training_cost_analysis.py --model_config part2/deepseek_v3_config.json
```

---

## Part 3: Speculative Decoding (50 pts + 10 pts bonus)

Open `part3/PA3_Speculative_Decoding.ipynb`. You will implement a single-batch
speculative decoder using a small **draft** model to propose tokens that a
larger **target** model verifies in one batched forward pass.

Default model pair (public weights, fits any GPU with >=4 GB VRAM):

- target: `EleutherAI/pythia-1.4b-deduped`
- draft:  `EleutherAI/pythia-160m-deduped`

If you swap to a different target/draft pair, document your measured
acceptance rate and the resulting speedup in your report.

### 3.1 Implementation (25 pts)

Fill in the notebook stubs:

- `initialize_target_model` and `initialize_draft_model` (5 pts)
- `generate_draft_tokens` (5 pts)
- `verify_tokens_vectorized` (5 pts) — single forward pass through the target
- `speculative_decode` main loop (10 pts)

### 3.2 Performance (20 pts)

- ≥ 1.0× wall-clock speedup over baseline target-only decoding (10 pts)
- ≥ 75% draft-token acceptance rate (10 pts)

### 3.3 Analysis and Evaluation (5 pts)

- Sweep `num_speculative_tokens ∈ {2, 4, 8, 16}` and report acceptance rate
  and speedup at each setting.
- Document any optimizations you applied (e.g. KV cache reuse for the draft,
  greedy vs. sampling, fp16 vs. bf16) and their measured effect.
- Write a short report (≤ 2 pages) with these results as `part3/report.md`
  (or `part3/report.pdf`). It ships inside `handin.tar`; there is no separate
  upload.

### Bonus 3.B — Tree / Multi-branch Speculation (10 pts)

The single-branch verifier accepts at most one chain per round. Implement
**either** (a) tree speculation with a tree-attention mask so the target can
verify multiple candidate branches in one forward pass, **or** (b) n-gram
lookup decoding combined with the standard draft model. Report acceptance
rate, speedup, and a one-paragraph discussion of why your variant changes the
acceptance rate. Replicating EAGLE-2 or Medusa qualifies.

---

## Part 4: Essay — The Future of LLMs and AI (40 pts)

> Submit Part 4 **separately** as a PDF on Gradescope under **PA3 — Essay**.
> Do not include it in `handin.tar`. Part 4 is **individual**.

### 4.1 Argumentative Essay (40 pts)

Write a ~500-word (±10%) argumentative essay defending one **conviction
statement** about the next 24–36 months of LLM / AI systems. You may choose
one of the convictions below or propose your own.

**Conviction options:**

- *Junior software engineer roles collapse.* By 2027-12-31, US BLS-tracked
  employment of software developers with less than 3 years of experience
  falls ≥20% from the 2025 baseline, with AI coding agents
  (Claude Code / Cursor / Devin-class tools) cited as the primary cause in
  industry layoff disclosures.
- *On-device SLMs claim the consumer query.* By the end of 2027, ≥40% of
  consumer chat queries on a flagship US/Korea/China smartphone are served
  entirely on-device (no cloud roundtrip) by a 4B–8B parameter model.
- *Inference-time compute exceeds 80% of frontier-lab spend.* By Q4 2027,
  ≥80% of annual compute spend at OpenAI / Anthropic / Google DeepMind
  combined is allocated to post-training (RL / RLVR / SFT) and
  inference-time scaling (search, reasoning chains, agent loops) — not
  next-token pretraining.
- *The agent revenue line overtakes chat APIs.* Agentic / tool-use revenue
  for Anthropic, OpenAI, or Google DeepMind exceeds chat-completion revenue
  by FY2028.
- *Hardware-economics breakup.* NVIDIA's data-center revenue declines ≥30%
  YoY at some point in 2027 driven by inference-side specialization
  (custom silicon, MoE-friendly accelerators).

You can also propose your own — it must be **specific**, **time-bound**, and
**measurable**.

**Requirements:**

- ~500 words ±10%
- Clear thesis defending the conviction
- At least one specific, time-bound, measurable prediction
- At least one acknowledged counter-argument and rebuttal
- Cite sources (papers, financial filings, benchmark releases — not blog
  vibes alone)

**Grading rubric:**

- Precision of arguments (40%) — specific, measurable, time-bound predictions
  beat vague generalizations.
- Evidence and reasoning (30%) — cite real numbers from real sources.
- Counterargument handling (15%) — acknowledge a serious objection and
  address it.
- Writing quality (15%) — organization, tone, citations.

> Your grade does not depend on whether your prediction comes true; it
> depends on the quality of the argument.
