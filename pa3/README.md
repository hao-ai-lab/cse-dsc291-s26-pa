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

You can run a provisional grader locally to see your score before you
submit. From the `pa3/` directory:

```bash
python local_check.py                  # Part 1 + Part 2 + Part 3 structural
python local_check.py --with-gpu       # Also execute Part 3 notebook (needs CUDA)
```

Sample output:

```
Part 1: MoE: 50/50
Part 2: cost analysis (main): 30/30
Part 2: cost analysis (bonus): 25/25
Part 3: speculative decoding (main): 25/50    # 25 = 3.1 structural; rest needs --with-gpu
...
PROVISIONAL TOTAL (excluding 40-pt essay): 130/165
```

> ⚠️ **PROVISIONAL SCORE NOTICE.** `local_check.py` is a sanity tool, not the
> final grade. After you submit, the instructor runs a held-out private test
> suite that exercises your functions on **different inputs** than the ones
> in `local_check.py` (e.g. a different model config for Part 2, different
> `(batch, hidden, topk)` for Part 1). If your function actually computes
> from its arguments, those held-out tests pass and your final score
> matches `local_check`. If your function hardcodes the public expected
> values, the held-out tests deduct up to ~15 points.

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

Test:

```bash
mpirun -n 4 python part1/test_moe.py
```

> If your machine has fewer physical cores than the requested process count,
> use `mpirun --oversubscribe -n 4 python part1/test_moe.py`.

### 1.1 Tensor Parallel (20 pts)

Every rank holds a slice (column shard) of every expert. Each rank computes a
partial expert output for the whole batch; ranks then collectively assemble
the full output.

- `ShardedLinear` (10 pts): a column-sharded linear layer that returns the
  full output on every rank. Use either `Allreduce` over a zero-padded local
  output or `Allgather` along the column dim.
- `MoE_TP` (10 pts): full TP forward pass that uses `ShardedExpert` and the
  replicated router.

### 1.2 Expert Parallel (20 pts)

Each rank holds **one** expert in its entirety. After routing, tokens have to
be shipped to the rank that owns the expert they were assigned to, then
results have to be shipped back. This is the canonical use case for
all-to-all.

- `MoE_EP` (20 pts): EP forward pass, using `mpi.alltoall(...)` (the
  pickle-based collective; supports variable-sized buckets per destination).

> **Bonus path (+5 pts).** If you copy `myAlltoall` from your PA2
> `mpi_wrapper/comm.py` into the marked location in `part1/mpi_wrapper/comm.py`
> and route the EP all-to-all through it (zero-padded so sizes match), you
> earn the bonus. We will verify by inspection and by running your tests with
> `MPI4PY_RC_RECV_MPROBE=0`.

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
- **Forward FLOPs** of a single transformer layer in TFLOPs.
  **Use `sequence_length = config["max_position_embeddings"]` and
  `batch_size = 1`** as the grading convention. (The autograder uses these
  values; using different values will make your numbers off by orders of
  magnitude and fail the tolerance check.)
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
with hyperparameters that hit your optimal `N`.

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
- Submit a short PDF report (≤ 2 pages) with these results.

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
