"""Local sanity check for PA3 (student-facing).

Run from the `pa3/` directory:

    python local_check.py

WHAT THIS DOES — AND DOESN'T — CHECK

This is a small, deliberately limited sanity tool. It only auto-checks the
things that are 100% deterministic and unambiguous:

  * Part 1 (MoE): your moe.py runs under mpirun and produces the correct
    output shape, replicated across ranks, and non-trivial (not all-zeros).
  * Part 2: your `model_training_cost_analysis_llama` returns the exact
    Llama-3 8B parameter count, and `get_optimal_N_D_from_cost(5_000_000)`
    selects the correct GPU.

Everything else is graded by the course staff by inspection of your
submitted files / saved notebook output:

  * Part 2 FLOPs / peak memory / (N, D) / DeepSeek-V3 / moe.md
  * Part 3 (the whole speculative-decoding notebook + report)
  * Part 4 (the essay)

A green local_check does NOT mean full marks, and a item we don't
auto-check is NOT worth zero — it just means a human grades it. Run the
Part 3 notebook yourself, save its output, and write the analysis/report;
that is what the staff read.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent  # pa3/

LLAMA3_8B_EXPECTED_PARAMS = 8_030_261_248
BUDGET_5M_EXPECTED_GPU = "B200"


# ======================================================================
# Part 1 — deterministic: shape + cross-rank replication + non-trivial
# ======================================================================
_PART1_WORKER = r'''
"""Worker run under mpirun by local_check.py. Writes a JSON report."""
import json
import os
import numpy as np

from mpi_wrapper import mpi
from rng import register_rng
from moe import SimpleMoE, MoE_TP, MoE_EP


def _reset(rank):
    register_rng("expert", np.random.RandomState(0))
    register_rng("router", np.random.RandomState(0))
    register_rng("expert_with_rank", np.random.RandomState(rank + 100))


def main():
    rank = mpi.Get_rank()
    ws = mpi.Get_size()
    dim = 8 * ws

    if mpi.Get_rank() == 0:
        X = np.random.RandomState(42).randn(6, dim)
    else:
        X = None
    X = mpi.bcast(X, root=0)

    report = {"world_size": ws}

    _reset(rank)
    simple = SimpleMoE(dim, dim * 2, dim, num_experts=ws, topk=2)
    report["expected_shape"] = list(simple(X).shape)

    try:
        _reset(rank)
        out_tp = MoE_TP(dim, dim * 2, dim, num_experts=ws, topk=2)(X)
        report["tp_shape"] = list(out_tp.shape)
        g = mpi.allgather(out_tp)
        if rank == 0:
            report["tp_cross_rank_max_diff"] = max(
                float(np.abs(g[0] - g[r]).max()) for r in range(ws))
            report["tp_output_max_abs"] = float(np.abs(g[0]).max())
    except Exception as e:
        report["tp_error"] = repr(e)

    try:
        _reset(rank)
        out_ep = MoE_EP(dim, dim * 2, dim, num_experts=ws, topk=2)(X)
        report["ep_shape"] = list(out_ep.shape)
        g = mpi.allgather(out_ep)
        if rank == 0:
            report["ep_cross_rank_max_diff"] = max(
                float(np.abs(g[0] - g[r]).max()) for r in range(ws))
            report["ep_output_max_abs"] = float(np.abs(g[0]).max())
    except Exception as e:
        report["ep_error"] = repr(e)

    if rank == 0:
        with open(os.environ["PART1_REPORT"], "w") as f:
            json.dump(report, f)


if __name__ == "__main__":
    main()
'''


def _grade_part1(world_size=4):
    part1 = HERE / "part1"
    items = []
    if not part1.exists():
        return [("part1/ missing", 0, 50, "")]

    worker = part1 / "_local_check_worker.py"
    report_path = part1 / "_local_check_report.json"
    worker.write_text(_PART1_WORKER)
    env = os.environ.copy()
    env["PART1_REPORT"] = str(report_path)
    cmd = ["mpirun", "--oversubscribe", "-n", str(world_size),
           sys.executable, str(worker)]
    try:
        proc = subprocess.run(cmd, cwd=str(part1), capture_output=True,
                               text=True, timeout=120, env=env)
        report = json.loads(report_path.read_text()) if report_path.exists() else None
    except subprocess.TimeoutExpired:
        report, proc = None, None
    finally:
        # Always clean up — never let these end up in `make handin.tar`.
        worker.unlink(missing_ok=True)
        report_path.unlink(missing_ok=True)

    if report is None:
        tail = (proc.stderr[-800:] if proc else "timed out")
        return [("mpirun worker failed", 0, 50, tail)]

    exp = report.get("expected_shape")

    # 1.1 ShardedLinear shape (10)
    tp_max = report.get("tp_output_max_abs")
    if report.get("tp_shape") != exp:
        items.append(("1.1 ShardedLinear shape", 0, 10,
                      f"TP shape={report.get('tp_shape')} expected={exp} "
                      f"{report.get('tp_error', '')}"))
    elif not tp_max or tp_max < 1e-10:
        items.append(("1.1 ShardedLinear shape", 5, 10,
                      "shape OK but TP output all-zero (stub)"))
    else:
        items.append(("1.1 ShardedLinear shape", 10, 10, "shape OK, non-trivial"))

    # 1.1 MoE_TP forward — replicated across ranks (10)
    xr = report.get("tp_cross_rank_max_diff")
    if report.get("tp_error"):
        items.append(("1.1 MoE_TP forward", 0, 10, report["tp_error"]))
    elif xr is None:
        items.append(("1.1 MoE_TP forward", 0, 10, "no cross-rank data"))
    elif xr < 1e-9 and (tp_max or 0) > 1e-10:
        items.append(("1.1 MoE_TP forward", 10, 10,
                      f"replicated across ranks (diff {xr:.1e})"))
    elif xr < 1e-9:
        items.append(("1.1 MoE_TP forward", 5, 10, "replicated but all-zero (stub)"))
    else:
        items.append(("1.1 MoE_TP forward", 0, 10,
                      f"NOT replicated across ranks (diff {xr:.1e})"))

    # 1.2 MoE_EP (20): shape (5) + non-trivial (5) + replicated (10)
    ep_score, parts = 0, []
    ep_max = report.get("ep_output_max_abs")
    if report.get("ep_shape") == exp:
        ep_score += 5
        parts.append("shape OK")
    else:
        parts.append(f"shape={report.get('ep_shape')} exp={exp} "
                     f"{report.get('ep_error', '')}")
    if ep_max and ep_max > 1e-10:
        ep_score += 5
        parts.append("non-trivial")
    else:
        parts.append("all-zero (stub)")
    exr = report.get("ep_cross_rank_max_diff")
    if exr is not None and exr < 1e-9:
        ep_score += 10
        parts.append(f"replicated (diff {exr:.1e})")
    else:
        parts.append(f"NOT replicated (diff {exr})")
    items.append(("1.2 MoE_EP", ep_score, 20, " | ".join(parts)))

    # 1.3 benchmark.py runs (5)
    bench = part1 / "benchmark.py"
    if not bench.exists():
        items.append(("1.3 benchmark.py runs", 0, 5, "missing"))
    else:
        try:
            p = subprocess.run(["mpirun", "--oversubscribe", "-n", "4",
                                 sys.executable, "benchmark.py"],
                                cwd=str(part1), capture_output=True,
                                text=True, timeout=180)
            items.append(("1.3 benchmark.py runs",
                           5 if p.returncode == 0 else 0, 5,
                           "OK" if p.returncode == 0 else p.stderr[-200:]))
        except subprocess.TimeoutExpired:
            items.append(("1.3 benchmark.py runs", 0, 5, "timed out"))

    # 1.3 analysis.md present + filled (5)
    ana = part1 / "analysis.md"
    if not ana.exists():
        items.append(("1.3 analysis.md", 0, 5, "missing"))
    elif "Replace this placeholder" in ana.read_text():
        items.append(("1.3 analysis.md", 0, 5,
                       "still the placeholder — write your analysis"))
    elif len(ana.read_text().strip()) < 200:
        items.append(("1.3 analysis.md", 0, 5, "too short"))
    else:
        items.append(("1.3 analysis.md", 5, 5, "present"))

    return items


# ======================================================================
# Part 2 — only the two certain checks; print the rest for self-review
# ======================================================================
def _load_student_mod(path):
    spec = importlib.util.spec_from_file_location("student_cost", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _grade_part2():
    part2 = HERE / "part2"
    checks, info = [], []
    if not part2.exists():
        return [("part2/ missing", False, "")], []
    try:
        mod = _load_student_mod(part2 / "model_training_cost_analysis.py")
    except Exception as e:
        return [(f"import failed: {type(e).__name__}: {e}", False, "")], []

    # Certain check 1: Llama-3 8B exact parameter count.
    try:
        p, f, m = mod.model_training_cost_analysis_llama(
            str(part2 / "llama3_8b_config.json"))
        checks.append(("Llama-3 8B total_params == 8,030,261,248",
                       p == LLAMA3_8B_EXPECTED_PARAMS, f"got {p:,}"))
        info.append(f"Llama-3 8B: flops_layer_TF={f}, peak_memory_GB={m} "
                    "(staff-reviewed — sanity-check these yourself)")
    except Exception as e:
        checks.append(("Llama-3 8B analysis runs", False,
                        f"{type(e).__name__}: {e}"))

    # Certain check 2: $5M budget selects B200.
    try:
        N, D, F, gpu = mod.get_optimal_N_D_from_cost(5_000_000)
        checks.append((f'get_optimal_N_D_from_cost($5M) best_gpu == "{BUDGET_5M_EXPECTED_GPU}"',
                       gpu == BUDGET_5M_EXPECTED_GPU, f"got {gpu!r}"))
        info.append(f"$5M optimum: N={N:.3e}, D={D:.3e}, F={F:.3e} "
                    "(staff-reviewed)")
    except Exception as e:
        checks.append(("get_optimal_N_D_from_cost runs", False,
                        f"{type(e).__name__}: {e}"))

    # DeepSeek + moe.md: not auto-graded — just report status.
    try:
        dp, df, dm = mod.model_training_cost_analysis_deepseek(
            str(part2 / "deepseek_v3_config.json"))
        info.append(f"DeepSeek-V3 (bonus): params={dp:,}, flops={df}, "
                    f"mem={dm} (staff-reviewed; ~671B params expected)")
    except Exception as e:
        info.append(f"DeepSeek-V3 (bonus): not implemented / errored "
                    f"({type(e).__name__}) — fine, it is optional")
    moe_md = part2 / "moe.md"
    if not moe_md.exists() or "Replace this placeholder" in moe_md.read_text():
        info.append("moe.md: still the placeholder (bonus, staff-reviewed)")
    else:
        info.append(f"moe.md: present ({len(moe_md.read_text().strip())} chars, "
                    "staff-reviewed)")
    return checks, info


# ======================================================================
# Reporting
# ======================================================================
def main():
    argparse.ArgumentParser(description=__doc__).parse_args()

    print("=" * 64)
    print("PA3 local sanity check — auto-checks only the certain items.")
    print("Parts 2 (FLOPs/mem/N,D/DeepSeek), 3, and 4 are graded by the")
    print("course staff from your submitted files. This is NOT your grade.")
    print("=" * 64)

    p1 = _grade_part1()
    print("\n[Part 1 — MoE]  (deterministic: shape + cross-rank + non-trivial)")
    p1_score = p1_max = 0
    for name, s, mx, detail in p1:
        p1_score += s
        p1_max += mx
        mark = "OK " if s == mx else " . " if s > 0 else "XX "
        print(f"  [{mark}] {name:<28} {s:>2}/{mx}  {detail}")
    print(f"  Part 1 deterministic subtotal: {p1_score}/{p1_max}")

    checks, info = _grade_part2()
    print("\n[Part 2 — certain checks only]")
    for name, ok, detail in checks:
        print(f"  [{'OK ' if ok else 'XX '}] {name}  ({detail})")
    print("\n[Part 2 — reported for your own sanity-check, staff-graded]")
    for line in info:
        print(f"  - {line}")

    print("\n[Part 3 — speculative decoding]")
    print("  Not auto-checked. Run the notebook on a GPU, SAVE its output")
    print("  (Speedup / acceptance lines), and write part3/report.md. The")
    print("  staff grade Part 3 from your saved notebook + report.")

    print("\n[Part 4 — essay]  Submitted separately as a PDF; graded by hand.")
    print("\n" + "=" * 64)
    print("Reminder: green here ≠ full marks; unchecked ≠ zero.")
    print("=" * 64)


if __name__ == "__main__":
    main()
