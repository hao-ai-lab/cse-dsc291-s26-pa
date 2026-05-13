# Part 1.3 — Benchmark Analysis

Replace this placeholder with your write-up.

Suggested structure:

1. **Setup.** Hardware (CPU model, core count), MPI library + version, world
   size, dtype.
2. **Sweeps.** What you varied (batch size, hidden dim, num experts, topk)
   and the configurations you measured. Include a table of timings (ms / fwd
   pass) for `SimpleMoE`, `MoE_TP`, and `MoE_EP`.
3. **Discussion.** Which variant is faster, and why? Identify whether the
   bottleneck for each variant is computation (matmul) or communication
   (which collective and how many bytes per rank). Tie the explanation back
   to the role of `Allreduce` / `Allgather` in TP and `alltoall` in EP.

Plots are optional but encouraged.
