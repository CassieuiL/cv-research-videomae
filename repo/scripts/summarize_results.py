#!/usr/bin/env python3
"""
summarize_results.py
扫描 results/tables/ 下所有 CSV，输出格式化的汇总表。
用法：python repo/scripts/summarize_results.py
"""

import csv
import os
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "results" / "tables"

def load_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def print_table(title, rows, cols):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")
    if not rows:
        print("  (no data)")
        return
    # Column widths
    widths = {c: max(len(c), max(len(str(r.get(c, ""))) for r in rows)) for c in cols}
    header = "  " + "  ".join(c.ljust(widths[c]) for c in cols)
    sep    = "  " + "  ".join("-" * widths[c] for c in cols)
    print(header)
    print(sep)
    for r in rows:
        print("  " + "  ".join(str(r.get(c, "")).ljust(widths[c]) for c in cols))

def main():
    print("\nVideoMAE UCF101 Ablation Study — Results Summary")
    print("=" * 70)

    # ── 1. All results ──────────────────────────────────────────────────────
    all_path = RESULTS_DIR / "all_results.csv"
    if all_path.exists():
        rows = load_csv(all_path)
        print_table(
            "FULL RESULTS TABLE",
            rows,
            ["week", "pretrain", "T", "stride", "inference_clips", "top1_acc", "notes"]
        )

    # ── 2. T ablation ───────────────────────────────────────────────────────
    t_path = RESULTS_DIR / "ablation_T.csv"
    if t_path.exists():
        rows = load_csv(t_path)
        print_table(
            "ABLATION: num_frames T  (stride=4, 1-clip)",
            rows,
            ["T", "epochs", "batch_size", "top1_acc", "delta_vs_T16", "notes"]
        )

    # ── 3. Stride ablation ──────────────────────────────────────────────────
    s_path = RESULTS_DIR / "ablation_stride.csv"
    if s_path.exists():
        rows = load_csv(s_path)
        print_table(
            "ABLATION: temporal stride  (T=16, 1-clip)",
            rows,
            ["stride", "epochs", "top1_acc", "delta_vs_stride4", "temporal_window_frames"]
        )

    # ── 4. Inference clips ablation ─────────────────────────────────────────
    i_path = RESULTS_DIR / "ablation_infer.csv"
    if i_path.exists():
        rows = load_csv(i_path)
        print_table(
            "ABLATION: inference clips  (T=16, stride=4)",
            rows,
            ["inference_clips", "top1_acc", "delta_vs_1clip", "approx_inference_cost_multiplier"]
        )

    # ── 5. Domain shift ─────────────────────────────────────────────────────
    d_path = RESULTS_DIR / "domain_shift.csv"
    if d_path.exists():
        rows = load_csv(d_path)
        print_table(
            "DOMAIN SHIFT: K400 vs SSV2 pretrain  (T=16, stride=4, 1-clip)",
            rows,
            ["pretrain_source", "domain_type", "top1_acc", "delta_vs_k400"]
        )

    # ── 6. Key findings ─────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  KEY FINDINGS")
    print(f"{'='*70}")
    findings = [
        "Best config (single vGPU-32GB): T=16, stride=4, K400, 5-clip → 84.67%",
        "T=16 optimal; T=8 drops -18.76%; T=32 OOM (batch 8→2, accuracy collapses)",
        "stride=4 >> stride=2 (+17.68%): UCF101 macro-actions need broad temporal coverage",
        "5-clip inference: +1.27% accuracy at ~5× inference cost",
        "Domain shift: K400→UCF101 outperforms SSV2→UCF101 by 7.8pp (domain similarity matters)",
    ]
    for i, f in enumerate(findings, 1):
        print(f"  {i}. {f}")
    print()

if __name__ == "__main__":
    main()
