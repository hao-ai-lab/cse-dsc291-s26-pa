# CSE 291 / DSC 291: Machine Learning Systems

This repository contains programming assignments for `CSE 291 / DSC 291` in `Spring 2026`.

Current contents:

- [`pa1/CSE291-S26-PA1.ipynb`](pa1/CSE291-S26-PA1.ipynb): PA1 notebook and assignment writeup
- [`pa1/auto_diff.py`](pa1/auto_diff.py): autodiff implementation scaffold
- [`pa1/transformer.py`](pa1/transformer.py): transformer training and generation code
- [`pa1/tests`](pa1/tests): public tests
- [`pa2/README.md`](pa2/README.md): PA2 assignment writeup (Triton matmul kernel + MPI tensor parallel)
- [`pa2/student_kernel.py`](pa2/student_kernel.py): Triton kernel scaffold (Part 1)
- [`pa2/mpi_wrapper`](pa2/mpi_wrapper): `comm.py` scaffold for `myAllreduce` / `myAlltoall` (Part 2)
- [`pa2/model`](pa2/model), [`pa2/data`](pa2/data): tensor/data parallel scaffolds (Part 2)
- [`pa2/tests`](pa2/tests): public tests
- [`pa3/README.md`](pa3/README.md): PA3 assignment writeup (MoE TP/EP, scaling-law cost analysis, speculative decoding, AI-future essay)
- [`pa3/part1`](pa3/part1): MoE scaffolds (`moe.py`, `mpi_wrapper/`, `benchmark.py`)
- [`pa3/part2`](pa3/part2): cost analysis scaffolds (`model_training_cost_analysis.py`, Llama-3 8B and DeepSeek-V3 configs)
- [`pa3/part3`](pa3/part3): speculative decoding notebook
