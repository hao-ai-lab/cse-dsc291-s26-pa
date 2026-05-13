"""Local self-check for PA3 (student-facing).

Run this from the `pa3/` directory:

    python local_check.py                  # Part 1 + Part 2 + Part 3 structural
    python local_check.py --with-gpu       # Also run the Part 3 notebook on a GPU

PROVISIONAL SCORE NOTICE:
    This is a *provisional* score. After you submit, the instructor runs a
    held-out private test suite against your handin to catch heavily
    hardcoded shortcuts. Your final score may differ from what local_check
    reports by up to ~10%. If you implement the assignment honestly (your
    function actually computes from the config / input, instead of just
    returning the expected magic number), the held-out tests will pass too.

What's covered:

    Part 1 (50): runs your moe.py under mpirun -n 4 and verifies that
                 (a) your MoE_TP forward matches your *own* SimpleMoE
                 numerically (oracle-via-self-consistency), and
                 (b) your MoE_EP forward is replicated across ranks.

    Part 2 (30 + 25 bonus): imports your model_training_cost_analysis.py
                 and checks the returned numbers against hard-coded expected
                 values + tolerances. The expected numbers are derivable
                 from the problem statement, so they're not "the answer";
                 returning them without computing them will fail the
                 held-out tests.

    Part 3 (50 + 10 bonus): parses your notebook for the four required
                 methods and (with --with-gpu) executes it to extract
                 speedup + acceptance rate from the benchmark cell.

Part 4 (essay, 40 pts) is graded manually.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path


HERE = Path(__file__).resolve().parent  # pa3/


# =====================================================================
# Part 1: MoE
# =====================================================================
_PART1_WORKER = r'''
"""Worker run inside mpirun -n 4 from local_check.py. Writes a JSON report."""
import json
import os
import sys
import numpy as np

from mpi_wrapper import mpi
from rng import register_rng
from moe import SimpleMoE, MoE_TP, MoE_EP


def _reset_rngs(rank):
    register_rng("expert", np.random.RandomState(0))
    register_rng("router", np.random.RandomState(0))
    register_rng("expert_with_rank", np.random.RandomState(rank + 100))


def _broadcast(seed, batch, dim):
    if mpi.Get_rank() == 0:
        X = np.random.RandomState(seed).randn(batch, dim)
    else:
        X = None
    return mpi.bcast(X, root=0)


def main():
    rank = mpi.Get_rank()
    ws = mpi.Get_size()
    dim = 8 * ws

    report = {"world_size": ws}

    # Draw the input ONCE, before any model construction, so the rng state
    # at SimpleMoE init matches the rng state at MoE_TP init exactly.
    X = _broadcast(42, 6, dim)

    # ----- SimpleMoE oracle -----
    _reset_rngs(rank)
    simple = SimpleMoE(dim, dim * 2, dim, num_experts=ws, topk=2)
    out_simple = simple(X)
    report["simple_shape"] = list(out_simple.shape)

    # ----- TP -----
    try:
        _reset_rngs(rank)
        tp = MoE_TP(dim, dim * 2, dim, num_experts=ws, topk=2)
        # Sanity-check that the router weights match SimpleMoE's, since both
        # were constructed under the same rng state.
        router_match = bool(np.allclose(simple.router.linear.weight, tp.router.linear.weight))
        report["tp_router_match_simple"] = router_match
        out_tp = tp(X)
        report["tp_shape"] = list(out_tp.shape)
        report["tp_vs_simple_max_diff"] = float(np.abs(out_simple - out_tp).max())
        report["tp_output_max_abs"] = float(np.abs(out_tp).max())
    except Exception as e:
        report["tp_error"] = repr(e)

    # ----- EP -----
    try:
        _reset_rngs(rank)
        ep = MoE_EP(dim, dim * 2, dim, num_experts=ws, topk=2)
        out_ep = ep(X)
        report["ep_shape"] = list(out_ep.shape)
        gathered = mpi.allgather(out_ep)
        if rank == 0:
            diffs = [float(np.abs(gathered[0] - gathered[r]).max()) for r in range(ws)]
            report["ep_cross_rank_max_diff"] = max(diffs)
            # Magnitude check: reject all-zero outputs (rejects the empty-stub
            # case that would otherwise pass cross-rank consistency trivially).
            report["ep_output_max_abs"] = float(np.abs(gathered[0]).max())
            # Order-of-magnitude sanity vs SimpleMoE: EP uses rank-specific
            # expert weights so values won't equal SimpleMoE, but they should
            # at least be on the same order of magnitude.
            simple_mag = float(np.abs(out_simple).max()) + 1e-12
            report["ep_vs_simple_log_ratio"] = float(
                np.log10(max(report["ep_output_max_abs"], 1e-30) / simple_mag)
            )
    except Exception as e:
        report["ep_error"] = repr(e)

    if rank == 0:
        with open(os.environ["PART1_REPORT"], "w") as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
'''


def _grade_part1(world_size: int = 4):
    part1 = HERE / "part1"
    if not part1.exists():
        return _section("Part 1: MoE", 0, 50, [("part1/ missing", 0, 50, "")])

    worker = part1 / "_local_check_worker.py"
    worker.write_text(_PART1_WORKER)
    report_path = part1 / "_local_check_report.json"
    if report_path.exists():
        report_path.unlink()

    cmd = ["mpirun", "--oversubscribe", "-n", str(world_size), sys.executable, str(worker)]
    env = os.environ.copy()
    env["PART1_REPORT"] = str(report_path)

    try:
        proc = subprocess.run(cmd, cwd=str(part1), capture_output=True, text=True,
                              timeout=120, env=env)
    except subprocess.TimeoutExpired:
        return _section("Part 1: MoE", 0, 50,
                        [("mpirun timeout", 0, 50, "your code took > 120s to run")])

    if not report_path.exists():
        stderr_tail = (proc.stderr or "")[-1200:]
        return _section("Part 1: MoE", 0, 50,
                        [("mpirun worker failed", 0, 50, f"rc={proc.returncode}\n{stderr_tail}")])

    with open(report_path) as f:
        report = json.load(f)
    report_path.unlink()

    items = []

    # 10 pts: ShardedLinear / TP shape (+ non-trivial output to reject stubs)
    tp_max = report.get("tp_output_max_abs")
    if report.get("tp_shape") != report.get("simple_shape"):
        items.append(("1.1 ShardedLinear shape", 0, 10,
                      f"TP shape={report.get('tp_shape')} simple={report.get('simple_shape')} "
                      f"err={report.get('tp_error', '')}"))
    elif tp_max is None or tp_max < 1e-10:
        items.append(("1.1 ShardedLinear shape", 5, 10,
                      f"shape OK but TP output is all-zero (max={tp_max}) — stub detected"))
    else:
        items.append(("1.1 ShardedLinear shape", 10, 10,
                      f"TP shape {report['tp_shape']} OK, output non-trivial"))

    # 10 pts: TP numerical match against student's own SimpleMoE
    diff = report.get("tp_vs_simple_max_diff")
    if diff is not None and diff < 1e-6:
        items.append(("1.1 MoE_TP numerical", 10, 10, f"max|TP - SimpleMoE| = {diff:.2e}"))
    elif diff is not None and diff < 1e-3:
        items.append(("1.1 MoE_TP numerical", 5, 10, f"max diff = {diff:.2e} (loose)"))
    else:
        items.append(("1.1 MoE_TP numerical", 0, 10, f"max diff = {diff}"))

    # 20 pts: EP shape (5) + non-trivial output (5) + cross-rank replicated (10)
    ep_score = 0
    ep_parts = []
    if report.get("ep_shape") == report.get("simple_shape"):
        ep_score += 5
        ep_parts.append(f"shape {report['ep_shape']} OK")
    else:
        ep_parts.append(f"shape MISMATCH ep={report.get('ep_shape')} "
                        f"expected={report.get('simple_shape')} "
                        f"err={report.get('ep_error', '')}")
    out_max = report.get("ep_output_max_abs")
    log_ratio = report.get("ep_vs_simple_log_ratio")
    if out_max is not None and out_max > 1e-10 and log_ratio is not None and abs(log_ratio) < 2:
        ep_score += 5
        ep_parts.append(f"non-trivial output max={out_max:.3e}")
    elif out_max is not None and out_max <= 1e-10:
        ep_parts.append(f"output is all-zero (max={out_max:.0e}) — stub detected")
    elif log_ratio is not None and abs(log_ratio) >= 2:
        ep_parts.append(f"output magnitude off by 100x (log_ratio={log_ratio:+.2f}) vs SimpleMoE")
    else:
        ep_parts.append("non-trivial check unavailable")
    diff_ep = report.get("ep_cross_rank_max_diff")
    if diff_ep is not None and diff_ep < 1e-6:
        ep_score += 10
        ep_parts.append(f"cross-rank diff = {diff_ep:.2e}")
    else:
        ep_parts.append(f"NOT replicated (diff={diff_ep})")
    items.append(("1.2 MoE_EP", ep_score, 20, " | ".join(ep_parts)))

    # 5 pts: benchmark.py runs
    bench = part1 / "benchmark.py"
    if not bench.exists():
        items.append(("1.3 benchmark.py runs", 0, 5, "benchmark.py missing"))
    else:
        try:
            p = subprocess.run(
                ["mpirun", "--oversubscribe", "-n", "4", sys.executable, "benchmark.py"],
                cwd=str(part1), capture_output=True, text=True, timeout=180)
            if p.returncode == 0:
                items.append(("1.3 benchmark.py runs", 5, 5, "OK"))
            else:
                items.append(("1.3 benchmark.py runs", 0, 5,
                              f"rc={p.returncode} {p.stderr[-300:]}"))
        except subprocess.TimeoutExpired:
            items.append(("1.3 benchmark.py runs", 0, 5, "timed out"))

    # 5 pts: analysis.md non-trivial
    ana = part1 / "analysis.md"
    if not ana.exists():
        items.append(("1.3 analysis.md", 0, 5, "missing"))
    elif "Replace this placeholder" in ana.read_text():
        items.append(("1.3 analysis.md", 0, 5,
                      "still contains the 'Replace this placeholder' marker — "
                      "delete the placeholder text and write your benchmark analysis"))
    else:
        text = ana.read_text()
        if len(text.strip()) < 200:
            items.append(("1.3 analysis.md", 0, 5, f"too short ({len(text.strip())} chars)"))
        else:
            items.append(("1.3 analysis.md", 5, 5, f"{len(text.strip())} chars"))

    return _section("Part 1: MoE", sum(s for _, s, _, _ in items), 50, items)


# =====================================================================
# Part 2: cost analysis (hard-coded expected values)
# =====================================================================

# These are all computed from the public assignment specification (configs +
# scaling law + GPU price table). Returning these numbers without actually
# implementing the function will fail the held-out tests run by the
# instructor after submission.
EXPECTED_LLAMA_PARAMS = 8_030_261_248
EXPECTED_LLAMA_FLOPS_TF_RANGE = (4.20, 5.15)   # +/-10% around 4.67
EXPECTED_LLAMA_MEMORY_GB_RANGE = (6.40, 9.60)  # +/-20% around 7.97

EXPECTED_BUDGET_5M_BEST_GPU = "B200"
EXPECTED_BUDGET_5M_FLOPS_RANGE = (2.673e24, 2.727e24)  # +/-1%
EXPECTED_BUDGET_5M_N_RANGE = (8.31e10, 1.124e11)        # +/-15%
EXPECTED_BUDGET_5M_D_RANGE = (3.91e12, 5.29e12)         # +/-15%

EXPECTED_DEEPSEEK_PARAMS_RANGE = (664_316_022_870, 677_736_535_978)  # +/-1%


def _load_module(path):
    spec = importlib.util.spec_from_file_location("_student_mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _safe(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs), None
    except NotImplementedError:
        return None, "NotImplementedError"
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def _in_range(v, lo_hi):
    lo, hi = lo_hi
    return v is not None and math.isfinite(v) and lo <= v <= hi


def _grade_part2():
    part2 = HERE / "part2"
    if not part2.exists():
        return (
            _section("Part 2: cost analysis (main)", 0, 30, [("part2/ missing", 0, 30, "")]),
            _section("Part 2: cost analysis (bonus)", 0, 25, [("part2/ missing", 0, 25, "")]),
        )

    try:
        mod = _load_module(part2 / "model_training_cost_analysis.py")
    except Exception as e:
        return (
            _section("Part 2: cost analysis (main)", 0, 30,
                     [("import failed", 0, 30, f"{type(e).__name__}: {e}")]),
            _section("Part 2: cost analysis (bonus)", 0, 25,
                     [("import failed", 0, 25, "")]),
        )

    main = []
    bonus = []

    # 2.1 Llama-3 8B (15 pts)
    cfg = part2 / "llama3_8b_config.json"
    out, err = _safe(mod.model_training_cost_analysis_llama, str(cfg))
    if out is None:
        main.append(("2.1 llama", 0, 15, f"call failed: {err}"))
    else:
        sp, sf, sm = out
        if sp == EXPECTED_LLAMA_PARAMS:
            main.append(("2.1 params", 8, 8, f"{sp:,}"))
        else:
            main.append(("2.1 params", 0, 8, f"got {sp}, expected {EXPECTED_LLAMA_PARAMS:,}"))
        if _in_range(sf, EXPECTED_LLAMA_FLOPS_TF_RANGE):
            main.append(("2.1 flops_TF", 4, 4, f"{sf:.2f}"))
        else:
            main.append(("2.1 flops_TF", 0, 4,
                         f"{sf} not in {EXPECTED_LLAMA_FLOPS_TF_RANGE} "
                         "(use seq_len = max_position_embeddings = 8192)"))
        if _in_range(sm, EXPECTED_LLAMA_MEMORY_GB_RANGE):
            main.append(("2.1 memory_GB", 3, 3, f"{sm:.2f}"))
        else:
            main.append(("2.1 memory_GB", 0, 3,
                         f"{sm} not in {EXPECTED_LLAMA_MEMORY_GB_RANGE}"))

    # 2.2 budget optimizer at $5M (15 pts)
    out, err = _safe(mod.get_optimal_N_D_from_cost, 5_000_000)
    if out is None:
        main.append(("2.2 budget", 0, 15, f"call failed: {err}"))
    else:
        sN, sD, sF, sGPU = out
        if sGPU == EXPECTED_BUDGET_5M_BEST_GPU:
            main.append(("2.2 best_gpu", 5, 5, sGPU))
        else:
            main.append(("2.2 best_gpu", 0, 5, f"got {sGPU}, expected {EXPECTED_BUDGET_5M_BEST_GPU}"))
        if _in_range(sF, EXPECTED_BUDGET_5M_FLOPS_RANGE):
            main.append(("2.2 budget_flops", 3, 3, f"{sF:.3e}"))
        else:
            main.append(("2.2 budget_flops", 0, 3, f"{sF} not in {EXPECTED_BUDGET_5M_FLOPS_RANGE}"))
        if _in_range(sN, EXPECTED_BUDGET_5M_N_RANGE):
            main.append(("2.2 optimal_N", 4, 4, f"{sN:.3e}"))
        else:
            main.append(("2.2 optimal_N", 0, 4, f"{sN} not in {EXPECTED_BUDGET_5M_N_RANGE}"))
        if _in_range(sD, EXPECTED_BUDGET_5M_D_RANGE):
            main.append(("2.2 optimal_D", 3, 3, f"{sD:.3e}"))
        else:
            main.append(("2.2 optimal_D", 0, 3, f"{sD} not in {EXPECTED_BUDGET_5M_D_RANGE}"))

    # 2.3 DeepSeek (15 bonus)
    cfg = part2 / "deepseek_v3_config.json"
    out, err = _safe(mod.model_training_cost_analysis_deepseek, str(cfg))
    if out is None:
        bonus.append(("2.3 deepseek", 0, 15, f"call failed: {err}"))
    else:
        sp, sf, sm = out
        if _in_range(sp, EXPECTED_DEEPSEEK_PARAMS_RANGE):
            bonus.append(("2.3 params", 8, 8, f"{sp:,}"))
        else:
            bonus.append(("2.3 params", 0, 8, f"{sp} not in {EXPECTED_DEEPSEEK_PARAMS_RANGE}"))
        bonus.append(("2.3 flops finite", 4 if (sf and math.isfinite(sf) and sf > 0) else 0,
                      4, f"{sf}"))
        bonus.append(("2.3 memory finite", 3 if (sm and math.isfinite(sm) and sm > 0) else 0,
                      3, f"{sm}"))

    # 2.3 moe.md (10 bonus)
    moe_md = part2 / "moe.md"
    if not moe_md.exists():
        bonus.append(("2.3 moe.md", 0, 10, "missing"))
    elif "Replace this placeholder" in moe_md.read_text():
        bonus.append(("2.3 moe.md", 0, 10,
                      "still contains the 'Replace this placeholder' marker — "
                      "delete the placeholder text and write your MoE discussion"))
    else:
        text = moe_md.read_text()
        has_activated = "activated" in text.lower() or "active" in text.lower()
        has_total = "total" in text.lower()
        if len(text.strip()) >= 300 and has_activated and has_total:
            bonus.append(("2.3 moe.md", 10, 10, f"{len(text.strip())} chars"))
        elif len(text.strip()) >= 300:
            bonus.append(("2.3 moe.md", 5, 10, "missing 'activated' or 'total' keywords"))
        else:
            bonus.append(("2.3 moe.md", 0, 10, f"too short ({len(text.strip())} chars)"))

    return (
        _section("Part 2: cost analysis (main)", sum(s for _, s, _, _ in main), 30, main),
        _section("Part 2: cost analysis (bonus)", sum(s for _, s, _, _ in bonus), 25, bonus),
    )


# =====================================================================
# Part 3: speculative decoding (structural + optional GPU exec)
# =====================================================================

_SPEEDUP_RE = re.compile(r"Speedup\s*[:=]\s*([\d.]+)x", re.IGNORECASE)
_ACCEPT_RE = re.compile(r"Draft\s+(?:token\s+)?acceptance\s+rate\s*[:=]\s*([\d.]+)%",
                        re.IGNORECASE)


def _grade_part3(with_gpu: bool, timeout: int = 1800):
    part3 = HERE / "part3"
    if not part3.exists():
        return _section("Part 3: speculative decoding (main)", 0, 50,
                        [("part3/ missing", 0, 50, "")]), \
               _section("Part 3: speculative decoding (bonus)", 0, 10, [])

    try:
        import nbformat
    except ImportError:
        return _section("Part 3: speculative decoding (main)", 0, 50,
                        [("nbformat not installed",
                          0, 50, "pip install nbformat (and nbclient if --with-gpu)")]), \
               _section("Part 3: speculative decoding (bonus)", 0, 10, [])

    # Find notebook
    nbs = sorted(part3.glob("*.ipynb"))
    nb_path = next((n for n in nbs if "spec" in n.name.lower()), nbs[0] if nbs else None)
    if nb_path is None:
        return _section("Part 3: speculative decoding (main)", 0, 50,
                        [("notebook missing", 0, 50, "")]), \
               _section("Part 3: speculative decoding (bonus)", 0, 10, [])

    nb = nbformat.read(nb_path, as_version=4)
    src = "\n".join(c.get("source", "") for c in nb.cells if c.get("cell_type") == "code")

    main = []
    # 3.1 structural (25)
    needed = {
        "initialize_target_model": (5, "3.1 init target+draft", "initialize_target_model"),
        "initialize_draft_model":  (None, None, None),
        "generate_draft_tokens":   (5, "3.1 generate_draft_tokens", None),
        "verify_tokens_vectorized":(5, "3.1 verify_tokens_vectorized", None),
        "speculative_decode":      (10, "3.1 speculative_decode", None),
    }
    has_init_t = f"def initialize_target_model" in src
    has_init_d = f"def initialize_draft_model" in src
    main.append(("3.1 init target+draft",
                 5 if (has_init_t and has_init_d) else 0, 5,
                 "both defined" if (has_init_t and has_init_d) else "missing one or both"))

    def _body(name):
        m = re.search(rf"def {name}\b.*?(?=\n    def |\nclass |\Z)", src, re.S)
        return m.group(0) if m else ""

    def _is_stub(body, required_substrings):
        """Return (is_stub, why). A method body is a stub if it still
        contains a `# TODO` marker, ends in `pass`, or has < 80 chars of
        actual non-comment code, or misses required API calls."""
        if not body:
            return True, "missing"
        if "# TODO" in body or "TODO (3.1)" in body:
            return True, "still contains '# TODO' marker"
        import re as _re
        stripped = _re.sub(r'""".*?"""', '', body, flags=_re.S)
        stripped = _re.sub(r"'''.*?'''", '', stripped, flags=_re.S)
        code_lines = [ln for ln in stripped.split("\n")
                      if ln.strip() and not ln.strip().startswith("#")]
        code_text = "\n".join(code_lines)
        if len(code_text.strip()) < 80:
            return True, f"only {len(code_text.strip())} chars of non-comment code"
        if code_text.strip().endswith("pass"):
            return True, "body ends in `pass`"
        missing = [s for s in required_substrings if s not in code_text]
        if missing:
            return True, f"missing required code: {missing}"
        return False, f"{len(code_text.strip())} chars of code"

    body = _body("generate_draft_tokens")
    is_stub, why = _is_stub(body, ["self.draft_model"])
    main.append(("3.1 generate_draft_tokens", 0 if is_stub else 5, 5, why))

    body = _body("verify_tokens_vectorized")
    is_stub, why = _is_stub(body, ["self.target_model", "argmax"])
    main.append(("3.1 verify_tokens_vectorized", 0 if is_stub else 5, 5, why))

    body = _body("speculative_decode")
    is_stub, why = _is_stub(
        body,
        ["self.verify_tokens_vectorized", "self.generate_draft_tokens"],
    )
    if is_stub:
        main.append(("3.1 speculative_decode", 0, 10, why))
    elif "while" in body or "for " in body:
        main.append(("3.1 speculative_decode", 10, 10, why + " + loop"))
    else:
        main.append(("3.1 speculative_decode", 4, 10, why + " — no loop?"))

    # 3.3 report (5)
    has_report = any((part3 / p).exists() and (part3 / p).stat().st_size > (5_000 if p.endswith(".pdf") else 400)
                     for p in ["report.pdf", "report.md", "writeup.pdf", "writeup.md", "REPORT.md"])
    main.append(("3.3 report", 5 if has_report else 0, 5,
                 "OK" if has_report else "no report.{pdf,md} found"))

    bonus_items = []
    bonus_keywords = ["tree", "n-gram", "ngram", "lookahead", "eagle", "medusa"]
    has_bonus_kw = any(k in src.lower() for k in bonus_keywords)

    if not with_gpu:
        main.append(("3.2 wall-clock + acceptance", 0, 20,
                     "skipped (run again with --with-gpu)"))
        bonus_items.append(("3.B bonus",
                            5 if has_bonus_kw else 0, 10,
                            "keywords found" if has_bonus_kw else "no bonus attempt"))
        return (
            _section("Part 3: speculative decoding (main)",
                     sum(s for _, s, _, _ in main), 50, main),
            _section("Part 3: speculative decoding (bonus)",
                     sum(s for _, s, _, _ in bonus_items), 10, bonus_items),
        )

    # Execute notebook
    try:
        from nbclient import NotebookClient
    except ImportError:
        main.append(("3.2 wall-clock + acceptance", 0, 20, "nbclient not installed"))
        bonus_items.append(("3.B bonus", 5 if has_bonus_kw else 0, 10, "kw-only score"))
        return (
            _section("Part 3: speculative decoding (main)",
                     sum(s for _, s, _, _ in main), 50, main),
            _section("Part 3: speculative decoding (bonus)",
                     sum(s for _, s, _, _ in bonus_items), 10, bonus_items),
        )

    print(f"  [Part 3] executing notebook (timeout {timeout}s) — this can take a few minutes...",
          file=sys.stderr)
    try:
        client = NotebookClient(
            nb, timeout=timeout, kernel_name="python3",
            resources={"metadata": {"path": str(nb_path.parent)}},
        )
        client.execute()
    except Exception as e:
        main.append(("3.2 wall-clock + acceptance", 0, 20,
                     f"notebook execute failed: {type(e).__name__}: {e}"))
        bonus_items.append(("3.B bonus", 5 if has_bonus_kw else 0, 10, "kw-only score"))
        return (
            _section("Part 3: speculative decoding (main)",
                     sum(s for _, s, _, _ in main), 50, main),
            _section("Part 3: speculative decoding (bonus)",
                     sum(s for _, s, _, _ in bonus_items), 10, bonus_items),
        )

    speedups, accepts = [], []
    for c in nb.cells:
        if c.get("cell_type") != "code":
            continue
        for o in c.get("outputs", []):
            text = o.get("text", "") or o.get("data", {}).get("text/plain", "")
            speedups.extend(float(m.group(1)) for m in _SPEEDUP_RE.finditer(text))
            accepts.extend(float(m.group(1)) / 100.0 for m in _ACCEPT_RE.finditer(text))

    if not speedups:
        main.append(("3.2 speedup", 0, 10, "no Speedup line found in notebook output"))
    else:
        ok = sum(1 for s in speedups if s >= 1.0)
        if ok >= 2 and len(speedups) >= 3:
            main.append(("3.2 speedup", 10, 10, f"{ok}/{len(speedups)} prompts >=1.0x: {speedups}"))
        elif ok >= 1:
            main.append(("3.2 speedup", 5, 10, f"only {ok}/{len(speedups)} hit >=1.0x"))
        else:
            main.append(("3.2 speedup", 0, 10, f"no prompt hit >=1.0x: {speedups}"))

    if not accepts:
        main.append(("3.2 acceptance", 0, 10, "no acceptance rate found"))
    else:
        avg = sum(accepts) / len(accepts)
        if avg >= 0.75:
            main.append(("3.2 acceptance", 10, 10, f"avg {avg:.2%}"))
        elif avg >= 0.50:
            main.append(("3.2 acceptance", 5, 10, f"avg {avg:.2%} (50-75%)"))
        else:
            main.append(("3.2 acceptance", 0, 10, f"avg only {avg:.2%}"))

    bonus_score = 0
    if has_bonus_kw and len(speedups) >= 4:  # main 3 + at least 1 from bonus
        bonus_score = 10
        bonus_detail = "bonus keywords + extra benchmark output"
    elif has_bonus_kw:
        bonus_score = 5
        bonus_detail = "bonus keywords but no extra benchmark"
    else:
        bonus_detail = "no bonus attempt"
    bonus_items.append(("3.B bonus", bonus_score, 10, bonus_detail))

    return (
        _section("Part 3: speculative decoding (main)",
                 sum(s for _, s, _, _ in main), 50, main),
        _section("Part 3: speculative decoding (bonus)",
                 sum(s for _, s, _, _ in bonus_items), 10, bonus_items),
    )


# =====================================================================
# Reporting
# =====================================================================
def _section(name, score, max_, items):
    return {"name": name, "score": score, "max": max_, "items": items}


def _print_section(s):
    print(f"\n{s['name']}: {s['score']}/{s['max']}")
    for name, score, max_, detail in s["items"]:
        mark = "OK " if score == max_ else " . " if score > 0 else "XX "
        print(f"  [{mark}] {name:<38} {score:>3}/{max_}   {detail}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--with-gpu", action="store_true",
                   help="Also execute the Part 3 notebook (requires CUDA GPU)")
    p.add_argument("--world-size", type=int, default=4)
    p.add_argument("--part3-timeout", type=int, default=1800)
    args = p.parse_args()

    print(textwrap.dedent("""
    ============================================================
    PA3 local check  (PROVISIONAL SCORE — see README for details)
    ============================================================
    Note: the instructor runs a held-out private test suite after
    submission to catch heavily hardcoded solutions. Your final
    score may differ from this by up to ~10%.
    """).strip())

    p1 = _grade_part1(world_size=args.world_size)
    p2_main, p2_bonus = _grade_part2()
    p3_main, p3_bonus = _grade_part3(with_gpu=args.with_gpu,
                                     timeout=args.part3_timeout)

    for s in (p1, p2_main, p2_bonus, p3_main, p3_bonus):
        _print_section(s)

    main_total = p1["score"] + p2_main["score"] + p3_main["score"]
    main_max = p1["max"] + p2_main["max"] + p3_main["max"]
    bonus_total = p2_bonus["score"] + p3_bonus["score"]
    bonus_max = p2_bonus["max"] + p3_bonus["max"]

    print()
    print("-" * 60)
    print(f"  AUTO MAIN: {main_total:>4}/{main_max}")
    print(f"  BONUS:     {bonus_total:>4}/{bonus_max}")
    print(f"  PROVISIONAL TOTAL (excluding 40-pt essay): "
          f"{main_total + bonus_total}/{main_max + bonus_max}")
    print("-" * 60)
    if not args.with_gpu:
        print("\nTip: run `python local_check.py --with-gpu` to also score Part 3.2 "
              "(wall-clock speedup + acceptance), which adds up to 20 main + 10 bonus points.")


if __name__ == "__main__":
    main()
