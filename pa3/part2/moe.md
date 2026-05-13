# Part 2.3 — Why MoE?

Replace this placeholder with your write-up.

Compare a dense Llama-3 8B against the activated-parameter footprint of
DeepSeek-V3 (~37B activated out of ~671B total). Discuss at least:

1. Total parameters vs. activated parameters per token, and how that shifts
   training FLOPs and memory.
2. Communication cost trade-offs (TP/EP collectives, all-to-all bandwidth) and
   how they scale with `num_experts` and `topk`.
3. Inference economics: why an MoE serves cheaper requests at low load but can
   become expensive at high load (token bucket imbalance, hot experts).
4. One concrete advantage and one concrete disadvantage relative to the dense
   baseline at the same training budget.
