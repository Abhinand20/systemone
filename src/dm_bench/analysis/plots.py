"""Matplotlib Agg chart generation for Milestone 1 (T1-T6, latency, Pareto, DM calibration)."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from dm_bench.analysis.stats import (
    CORE_ROW_ORDER,
    DEFAULT_BOOTSTRAP_SAMPLES,
    DEFAULT_BOOTSTRAP_SEED,
    RENDER_VARIANT_ORDER,
    TACTICAL_CATEGORY_ORDER,
    TEXT_FORMAT_ORDER,
    bootstrap_ci,
    compute_dm_calibration,
    compute_tactical_metrics,
    compute_track2_metrics,
    select_core_t2_decisions,
    select_dm_decisions,
    select_t3_decisions,
    select_t4_decisions,
    select_t5_decisions,
    sort_row_ids,
)
from dm_bench.contracts import DecisionLogRow, EpisodeLogRow

CHART_NAMES: tuple[str, ...] = (
    "t2_core_accuracy_illegal",
    "t1_tactical_heatmap",
    "t3_rendering_robustness",
    "t4_legality_schema_delta",
    "t5_prompt_formats",
    "t6_match_outcomes",
    "latency_cdf",
    "pareto_quality_vs_cost",
    "dm_calibration_and_abstention",
)

# Colorblind-safe palette
PALETTE = [
    "#1D4ED8",  # deep blue
    "#DC2626",  # crimson
    "#059669",  # emerald
    "#D97706",  # amber
    "#7C3AED",  # violet
    "#0891B2",  # cyan
    "#BE185D",  # magenta
    "#4B5563",  # slate
    "#65A30D",  # lime
]


@dataclass(frozen=True, slots=True)
class PlotArtifact:
    """Generated or skipped plot entry."""

    name: str
    png_path: Path | None
    svg_path: Path | None
    generated: bool
    reason: str | None = None


def _apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#D1D5DB",
            "axes.grid": True,
            "grid.color": "#E5E7EB",
            "grid.linestyle": "--",
            "grid.linewidth": 0.6,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.5,
        }
    )


def _save_fig(fig: plt.Figure, output_dir: Path, name: str) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{name}.png"
    svg_path = output_dir / f"{name}.svg"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    return png_path, svg_path


def _plot_t2_core_accuracy_illegal(
    decisions: Sequence[DecisionLogRow],
    output_dir: Path,
    *,
    n_bootstrap: int,
    seed: int,
) -> PlotArtifact:
    name = "t2_core_accuracy_illegal"
    core = select_core_t2_decisions(decisions)
    if not core:
        return PlotArtifact(name=name, png_path=None, svg_path=None, generated=False, reason="No T2 core decisions found")

    by_row: dict[str, list[DecisionLogRow]] = defaultdict(list)
    for row in core:
        by_row[row.row_id].append(row)

    rows = sort_row_ids(list(by_row.keys()))
    opt_means: list[float] = []
    opt_err_low: list[float] = []
    opt_err_high: list[float] = []
    ill_means: list[float] = []
    ill_err_low: list[float] = []
    ill_err_high: list[float] = []

    for rid in rows:
        group = by_row[rid]
        opt_vals = [1.0 if r.action_is_optimal else 0.0 for r in group]
        ill_vals = [1.0 if (r.response_valid and not r.action_is_legal) else 0.0 for r in group]
        opt_ci = bootstrap_ci(opt_vals, n_bootstrap=n_bootstrap, seed=seed)
        ill_ci = bootstrap_ci(ill_vals, n_bootstrap=n_bootstrap, seed=seed)
        opt_means.append(opt_ci.mean * 100.0)
        opt_err_low.append(max(0.0, (opt_ci.mean - opt_ci.ci_low) * 100.0))
        opt_err_high.append(max(0.0, (opt_ci.ci_high - opt_ci.mean) * 100.0))
        ill_means.append(ill_ci.mean * 100.0)
        ill_err_low.append(max(0.0, (ill_ci.mean - ill_ci.ci_low) * 100.0))
        ill_err_high.append(max(0.0, (ill_ci.ci_high - ill_ci.mean) * 100.0))

    x = np.arange(len(rows))
    width = 0.36
    fig, ax = plt.subplots(figsize=(max(8.5, len(rows) * 1.1), 5.0))
    try:
        b1 = ax.bar(
            x - width / 2,
            opt_means,
            width,
            yerr=[opt_err_low, opt_err_high],
            capsize=3,
            color="#1D4ED8",
            alpha=0.9,
            label="Optimal Accuracy (%)",
            zorder=3,
        )
        b2 = ax.bar(
            x + width / 2,
            ill_means,
            width,
            yerr=[ill_err_low, ill_err_high],
            capsize=3,
            color="#DC2626",
            alpha=0.85,
            label="Illegal Move Rate (%)",
            zorder=3,
        )
        for bar, val in zip(b1, opt_means, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + 2.0,
                f"{val:.1f}%",
                ha="center",
                va="bottom",
                fontsize=7.5,
                color="#1E3A8A",
            )
        for bar, val in zip(b2, ill_means, strict=True):
            if val > 0.0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    val + 2.0,
                    f"{val:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=7.5,
                    color="#991B1B",
                )
        ax.set_xticks(x)
        ax.set_xticklabels(rows, rotation=25, ha="right")
        ax.set_ylim(0, 112)
        ax.set_ylabel("Percentage (%)")
        ax.set_title("T2 Core Matrix: Optimal Accuracy & Illegal Move Rate (95% Bootstrap CI)")
        ax.legend(loc="upper right")
        png_path, svg_path = _save_fig(fig, output_dir, name)
    finally:
        plt.close(fig)

    return PlotArtifact(name=name, png_path=png_path, svg_path=svg_path, generated=True)


def _plot_t1_tactical_heatmap(
    decisions: Sequence[DecisionLogRow],
    output_dir: Path,
) -> PlotArtifact:
    name = "t1_tactical_heatmap"
    core = select_core_t2_decisions(decisions)
    pool = core if core else [
        d for d in decisions if d.experiment_id not in {"M0_smoke", "T6"}
    ]
    pool = [d for d in pool if d.tactical_category]
    if not pool:
        return PlotArtifact(name=name, png_path=None, svg_path=None, generated=False, reason="No tactical_category annotations found")

    tactical_stats = compute_tactical_metrics(pool)
    rows = sort_row_ids(list({k[0] for k in tactical_stats.keys()}))
    cats = [c for c in TACTICAL_CATEGORY_ORDER if any((r, c) in tactical_stats for r in rows)]
    extra_cats = sorted({k[1] for k in tactical_stats.keys()} - set(cats))
    cats.extend(extra_cats)
    if not rows or not cats:
        return PlotArtifact(name=name, png_path=None, svg_path=None, generated=False, reason="Insufficient tactical matrix cells")

    matrix = np.full((len(rows), len(cats)), np.nan, dtype=np.float64)
    counts = np.zeros((len(rows), len(cats)), dtype=int)
    for r_idx, rid in enumerate(rows):
        for c_idx, cat in enumerate(cats):
            m = tactical_stats.get((rid, cat))
            if m is not None:
                matrix[r_idx, c_idx] = m.optimal_accuracy * 100.0
                counts[r_idx, c_idx] = m.n

    fig, ax = plt.subplots(figsize=(max(8.0, len(cats) * 1.45), max(4.5, len(rows) * 0.55)))
    try:
        ax.grid(False)
        cmap = plt.get_cmap("YlGnBu").with_extremes(bad="#F3F4F6")
        im = ax.imshow(matrix, cmap=cmap, vmin=0.0, vmax=100.0, aspect="auto")
        cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
        cbar.set_label("Optimal Accuracy (%)")

        ax.set_xticks(np.arange(len(cats)))
        ax.set_xticklabels([c.replace("_", "\n") for c in cats])
        ax.set_yticks(np.arange(len(rows)))
        ax.set_yticklabels(rows)
        ax.set_title("T1 Tactical Breakdown: Optimal-Move Accuracy (%) by Category")

        for r_idx in range(len(rows)):
            for c_idx in range(len(cats)):
                val = matrix[r_idx, c_idx]
                if np.isnan(val):
                    ax.text(c_idx, r_idx, "N/A", ha="center", va="center", fontsize=8, color="#6B7280")
                else:
                    txt_color = "white" if val >= 60.0 else "#111827"
                    ax.text(
                        c_idx,
                        r_idx,
                        f"{val:.0f}%\n(n={counts[r_idx, c_idx]})",
                        ha="center",
                        va="center",
                        fontsize=7.5,
                        color=txt_color,
                    )
        png_path, svg_path = _save_fig(fig, output_dir, name)
    finally:
        plt.close(fig)

    return PlotArtifact(name=name, png_path=png_path, svg_path=svg_path, generated=True)


def _plot_t3_rendering_robustness(
    decisions: Sequence[DecisionLogRow],
    output_dir: Path,
    *,
    n_bootstrap: int,
    seed: int,
) -> PlotArtifact:
    name = "t3_rendering_robustness"
    t3 = select_t3_decisions(decisions)
    if not t3:
        return PlotArtifact(name=name, png_path=None, svg_path=None, generated=False, reason="No T3 rendering perturbation decisions found")

    grouped: dict[tuple[str, str], list[DecisionLogRow]] = defaultdict(list)
    for row in t3:
        variant = row.render_variant or "clean_default"
        grouped[(row.row_id, variant)].append(row)

    rows = sort_row_ids(list({k[0] for k in grouped.keys()}))
    present_variants = {k[1] for k in grouped.keys()}
    variants = [v for v in RENDER_VARIANT_ORDER if v in present_variants]
    variants.extend(sorted(present_variants - set(variants)))
    if not rows or not variants:
        return PlotArtifact(name=name, png_path=None, svg_path=None, generated=False, reason="No T3 row/variant pairs")

    x = np.arange(len(variants))
    n_rows = len(rows)
    width = min(0.25, 0.78 / max(1, n_rows))

    fig, ax = plt.subplots(figsize=(max(8.5, len(variants) * 1.6), 5.0))
    try:
        for idx, rid in enumerate(rows):
            means: list[float] = []
            err_lo: list[float] = []
            err_hi: list[float] = []
            for var in variants:
                cell = grouped.get((rid, var), [])
                if cell:
                    ci = bootstrap_ci(
                        [1.0 if r.action_is_optimal else 0.0 for r in cell],
                        n_bootstrap=n_bootstrap,
                        seed=seed,
                    )
                    means.append(ci.mean * 100.0)
                    err_lo.append(max(0.0, (ci.mean - ci.ci_low) * 100.0))
                    err_hi.append(max(0.0, (ci.ci_high - ci.mean) * 100.0))
                else:
                    means.append(0.0)
                    err_lo.append(0.0)
                    err_hi.append(0.0)
            offset = (idx - (n_rows - 1) / 2.0) * width
            ax.bar(
                x + offset,
                means,
                width,
                yerr=[err_lo, err_hi],
                capsize=3,
                color=PALETTE[idx % len(PALETTE)],
                alpha=0.9,
                label=rid,
                zorder=3,
            )

        ax.set_xticks(x)
        ax.set_xticklabels([v.replace("_", "\n") for v in variants])
        ax.set_ylim(0, 110)
        ax.set_ylabel("Optimal Accuracy (%)")
        ax.set_title("T3 Rendering Robustness Across Visual Perturbations (95% CI)")
        ax.legend(loc="upper right")
        png_path, svg_path = _save_fig(fig, output_dir, name)
    finally:
        plt.close(fig)

    return PlotArtifact(name=name, png_path=png_path, svg_path=svg_path, generated=True)


def _plot_t4_legality_schema_delta(
    decisions: Sequence[DecisionLogRow],
    output_dir: Path,
    *,
    n_bootstrap: int,
    seed: int,
) -> PlotArtifact:
    name = "t4_legality_schema_delta"
    all9_rows, legal_only_rows = select_t4_decisions(decisions)
    if not legal_only_rows or not all9_rows:
        return PlotArtifact(name=name, png_path=None, svg_path=None, generated=False, reason="Need both all_9 baseline and T4 legal_only decisions")

    by_row_all9: dict[str, list[DecisionLogRow]] = defaultdict(list)
    by_row_legal: dict[str, list[DecisionLogRow]] = defaultdict(list)
    for r in all9_rows:
        by_row_all9[r.row_id].append(r)
    for r in legal_only_rows:
        by_row_legal[r.row_id].append(r)

    rows = sort_row_ids(list(set(by_row_all9.keys()) & set(by_row_legal.keys())))
    if not rows:
        return PlotArtifact(name=name, png_path=None, svg_path=None, generated=False, reason="No overlapping rows between all_9 and legal_only")

    acc_all9: list[float] = []
    err_all9_lo: list[float] = []
    err_all9_hi: list[float] = []
    acc_legal: list[float] = []
    err_legal_lo: list[float] = []
    err_legal_hi: list[float] = []

    for rid in rows:
        c1 = bootstrap_ci(
            [1.0 if r.action_is_optimal else 0.0 for r in by_row_all9[rid]],
            n_bootstrap=n_bootstrap,
            seed=seed,
        )
        c2 = bootstrap_ci(
            [1.0 if r.action_is_optimal else 0.0 for r in by_row_legal[rid]],
            n_bootstrap=n_bootstrap,
            seed=seed,
        )
        acc_all9.append(c1.mean * 100.0)
        err_all9_lo.append(max(0.0, (c1.mean - c1.ci_low) * 100.0))
        err_all9_hi.append(max(0.0, (c1.ci_high - c1.mean) * 100.0))
        acc_legal.append(c2.mean * 100.0)
        err_legal_lo.append(max(0.0, (c2.mean - c2.ci_low) * 100.0))
        err_legal_hi.append(max(0.0, (c2.ci_high - c2.mean) * 100.0))

    x = np.arange(len(rows))
    width = 0.36
    fig, ax = plt.subplots(figsize=(max(8.5, len(rows) * 1.1), 5.0))
    try:
        ax.bar(
            x - width / 2,
            acc_all9,
            width,
            yerr=[err_all9_lo, err_all9_hi],
            capsize=3,
            color="#6B7280",
            alpha=0.88,
            label="all_9 (default)",
            zorder=3,
        )
        ax.bar(
            x + width / 2,
            acc_legal,
            width,
            yerr=[err_legal_lo, err_legal_hi],
            capsize=3,
            color="#059669",
            alpha=0.9,
            label="legal_only (T4)",
            zorder=3,
        )
        for i in range(len(rows)):
            delta = acc_legal[i] - acc_all9[i]
            top = max(acc_all9[i] + err_all9_hi[i], acc_legal[i] + err_legal_hi[i]) + 2.0
            sign = "+" if delta >= 0 else ""
            ax.text(
                x[i],
                min(106.0, top),
                f"{sign}{delta:.1f}pp",
                ha="center",
                va="bottom",
                fontsize=7.5,
                fontweight="bold",
                color="#065F46" if delta >= 0 else "#991B1B",
            )
        ax.set_xticks(x)
        ax.set_xticklabels(rows, rotation=25, ha="right")
        ax.set_ylim(0, 114)
        ax.set_ylabel("Optimal Accuracy (%)")
        ax.set_title("T4 Action-Choice Schema: all_9 vs legal_only Optimal Accuracy")
        ax.legend(loc="upper right")
        png_path, svg_path = _save_fig(fig, output_dir, name)
    finally:
        plt.close(fig)

    return PlotArtifact(name=name, png_path=png_path, svg_path=svg_path, generated=True)


def _plot_t5_prompt_formats(
    decisions: Sequence[DecisionLogRow],
    output_dir: Path,
    *,
    n_bootstrap: int,
    seed: int,
) -> PlotArtifact:
    name = "t5_prompt_formats"
    t5 = select_t5_decisions(decisions)
    if not t5:
        return PlotArtifact(name=name, png_path=None, svg_path=None, generated=False, reason="No T5 prompt format decisions found")

    grouped: dict[tuple[str, str], list[DecisionLogRow]] = defaultdict(list)
    for row in t5:
        fmt = row.text_format or "json_canonical"
        grouped[(row.row_id, fmt)].append(row)

    rows = sort_row_ids(list({k[0] for k in grouped.keys()}))
    present_fmts = {k[1] for k in grouped.keys()}
    formats = [f for f in TEXT_FORMAT_ORDER if f in present_fmts]
    formats.extend(sorted(present_fmts - set(formats)))
    if not rows or not formats:
        return PlotArtifact(name=name, png_path=None, svg_path=None, generated=False, reason="No T5 row/format pairs")

    x = np.arange(len(formats))
    n_rows = len(rows)
    width = min(0.25, 0.78 / max(1, n_rows))

    fig, ax = plt.subplots(figsize=(max(8.0, len(formats) * 1.8), 5.0))
    try:
        for idx, rid in enumerate(rows):
            means: list[float] = []
            err_lo: list[float] = []
            err_hi: list[float] = []
            for fmt in formats:
                cell = grouped.get((rid, fmt), [])
                if cell:
                    ci = bootstrap_ci(
                        [1.0 if r.action_is_optimal else 0.0 for r in cell],
                        n_bootstrap=n_bootstrap,
                        seed=seed,
                    )
                    means.append(ci.mean * 100.0)
                    err_lo.append(max(0.0, (ci.mean - ci.ci_low) * 100.0))
                    err_hi.append(max(0.0, (ci.ci_high - ci.mean) * 100.0))
                else:
                    means.append(0.0)
                    err_lo.append(0.0)
                    err_hi.append(0.0)
            offset = (idx - (n_rows - 1) / 2.0) * width
            ax.bar(
                x + offset,
                means,
                width,
                yerr=[err_lo, err_hi],
                capsize=3,
                color=PALETTE[idx % len(PALETTE)],
                alpha=0.9,
                label=rid,
                zorder=3,
            )

        ax.set_xticks(x)
        ax.set_xticklabels([f.replace("_", "\n") for f in formats])
        ax.set_ylim(0, 110)
        ax.set_ylabel("Optimal Accuracy (%)")
        ax.set_title("T5 Text Representation Sensitivity Across Prompt Formats (95% CI)")
        ax.legend(loc="upper right")
        png_path, svg_path = _save_fig(fig, output_dir, name)
    finally:
        plt.close(fig)

    return PlotArtifact(name=name, png_path=png_path, svg_path=svg_path, generated=True)


def _plot_t6_match_outcomes(
    episodes: Sequence[EpisodeLogRow],
    output_dir: Path,
) -> PlotArtifact:
    name = "t6_match_outcomes"
    if not episodes:
        return PlotArtifact(name=name, png_path=None, svg_path=None, generated=False, reason="No T6 episodes found")

    t2_metrics = compute_track2_metrics(episodes)
    modes = [m for m in ("all_9", "legal_only") if any(k[1] == m for k in t2_metrics)]
    opponents = [o for o in ("random", "minimax") if any(k[2] == o for k in t2_metrics)]
    if not modes or not opponents:
        return PlotArtifact(name=name, png_path=None, svg_path=None, generated=False, reason="No T6 opponent/mode groups")

    n_panels = len(modes)
    n_opps = len(opponents)
    fig, axes = plt.subplots(
        n_panels,
        n_opps,
        figsize=(max(6.5 * n_opps, 8.0), 4.6 * n_panels),
        squeeze=False,
    )
    try:
        for r_idx, mode in enumerate(modes):
            for c_idx, opp in enumerate(opponents):
                ax = axes[r_idx][c_idx]
                rows = sort_row_ids(
                    [k[0] for k in t2_metrics if k[1] == mode and k[2] == opp]
                )
                if not rows:
                    ax.set_visible(False)
                    continue
                wins = np.array([t2_metrics[(rid, mode, opp)].win_rate * 100.0 for rid in rows])
                draws = np.array([t2_metrics[(rid, mode, opp)].draw_rate * 100.0 for rid in rows])
                loss_b = np.array([t2_metrics[(rid, mode, opp)].loss_by_board_rate * 100.0 for rid in rows])
                loss_i = np.array([t2_metrics[(rid, mode, opp)].loss_by_illegal_move_rate * 100.0 for rid in rows])

                x = np.arange(len(rows))
                ax.bar(x, wins, color="#059669", label="Win", zorder=3)
                ax.bar(x, draws, bottom=wins, color="#3B82F6", label="Draw", zorder=3)
                ax.bar(x, loss_b, bottom=wins + draws, color="#F59E0B", label="Loss (Board)", zorder=3)
                ax.bar(
                    x,
                    loss_i,
                    bottom=wins + draws + loss_b,
                    color="#DC2626",
                    label="Loss (Illegal)",
                    zorder=3,
                )
                ax.set_xticks(x)
                ax.set_xticklabels(rows, rotation=28, ha="right")
                ax.set_ylim(0, 105)
                ax.set_ylabel("Match Outcome (%)")
                ax.set_title(f"T6 vs {opp.title()} ({mode})")
                if r_idx == 0 and c_idx == n_opps - 1:
                    ax.legend(loc="upper right", fontsize=8)

        fig.suptitle("T6 Closed-Loop Match Outcome Breakdown", fontsize=13, fontweight="bold", y=1.01)
        fig.tight_layout()
        png_path, svg_path = _save_fig(fig, output_dir, name)
    finally:
        plt.close(fig)

    return PlotArtifact(name=name, png_path=png_path, svg_path=svg_path, generated=True)


def _plot_latency_cdf(
    decisions: Sequence[DecisionLogRow],
    output_dir: Path,
) -> PlotArtifact:
    name = "latency_cdf"
    core = select_core_t2_decisions(decisions)
    pool = core if core else [d for d in decisions if d.experiment_id != "M0_smoke"]

    by_row: dict[str, list[float]] = defaultdict(list)
    for row in pool:
        lat = row.end_to_end_latency_ms if row.end_to_end_latency_ms is not None else row.model_latency_ms
        if lat is not None and lat >= 0:
            by_row[row.row_id].append(float(lat))

    if not by_row:
        return PlotArtifact(name=name, png_path=None, svg_path=None, generated=False, reason="No latency measurements found")

    rows = sort_row_ids(list(by_row.keys()))
    all_lats = [v for vals in by_row.values() for v in vals]
    min_pos = min((v for v in all_lats if v > 0), default=1.0)
    max_val = max(all_lats, default=1.0)
    use_log = max_val / max(min_pos, 1e-6) > 8.0

    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    try:
        for idx, rid in enumerate(rows):
            vals = np.sort(np.asarray(by_row[rid], dtype=np.float64))
            cdf = np.arange(1, len(vals) + 1) / len(vals)
            p50 = float(np.percentile(vals, 50))
            p95 = float(np.percentile(vals, 95))
            color = PALETTE[idx % len(PALETTE)]
            ax.step(
                vals,
                cdf,
                where="post",
                color=color,
                linewidth=1.8,
                label=f"{rid} (p50={p50:.0f}ms, p95={p95:.0f}ms)",
            )
            ax.scatter([p50, p95], [0.5, 0.95], color=color, s=22, zorder=4)

        if use_log:
            ax.set_xscale("log")
        ax.axhline(0.5, color="#9CA3AF", linestyle=":", linewidth=1.0)
        ax.axhline(0.95, color="#9CA3AF", linestyle="--", linewidth=1.0)
        ax.set_ylim(0.0, 1.03)
        ax.set_xlabel("End-to-End Latency (ms)")
        ax.set_ylabel("Empirical CDF")
        ax.set_title("Decision End-to-End Latency CDF (p50 & p95 Markers)")
        ax.legend(loc="lower right", fontsize=8)
        png_path, svg_path = _save_fig(fig, output_dir, name)
    finally:
        plt.close(fig)

    return PlotArtifact(name=name, png_path=png_path, svg_path=svg_path, generated=True)


def _plot_pareto_quality_vs_cost(
    decisions: Sequence[DecisionLogRow],
    output_dir: Path,
) -> PlotArtifact:
    name = "pareto_quality_vs_cost"
    core = select_core_t2_decisions(decisions)
    pool = core if core else [d for d in decisions if d.experiment_id != "M0_smoke"]
    if not pool:
        return PlotArtifact(name=name, png_path=None, svg_path=None, generated=False, reason="No Track 1 decisions for Pareto plot")

    by_row: dict[str, list[DecisionLogRow]] = defaultdict(list)
    for row in pool:
        by_row[row.row_id].append(row)

    rows = sort_row_ids(list(by_row.keys()))
    points: list[tuple[str, str, float, float]] = []
    for rid in rows:
        grp = by_row[rid]
        acc = (sum(1 for r in grp if r.action_is_optimal) / len(grp)) * 100.0
        cost_100 = (sum(r.api_cost_usd for r in grp) / len(grp)) * 100.0
        points.append((rid, grp[0].family, cost_100, acc))

    # Pareto frontier: minimize cost_100, maximize acc
    sorted_pts = sorted(points, key=lambda p: (p[2], -p[3]))
    frontier: list[tuple[float, float]] = []
    best_acc = -1.0
    for _, _, cost_100, acc in sorted_pts:
        if acc > best_acc + 1e-9:
            frontier.append((cost_100, acc))
            best_acc = acc

    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    try:
        for family, marker, color, label in (
            ("decision_model", "o", "#1D4ED8", "Decision Model"),
            ("generative", "^", "#DC2626", "Generative VLM/LLM"),
        ):
            fam_pts = [p for p in points if p[1] == family]
            if fam_pts:
                ax.scatter(
                    [p[2] for p in fam_pts],
                    [p[3] for p in fam_pts],
                    s=85,
                    marker=marker,
                    color=color,
                    label=label,
                    zorder=4,
                )
        other_pts = [p for p in points if p[1] not in ("decision_model", "generative")]
        if other_pts:
            ax.scatter(
                [p[2] for p in other_pts],
                [p[3] for p in other_pts],
                s=85,
                marker="s",
                color="#059669",
                label="Other",
                zorder=4,
            )

        for rid, _, cost_100, acc in points:
            ax.annotate(
                rid,
                (cost_100, acc),
                textcoords="offset points",
                xytext=(6, 4),
                fontsize=8,
            )

        if len(frontier) >= 2:
            ax.plot(
                [p[0] for p in frontier],
                [p[1] for p in frontier],
                linestyle="--",
                color="#059669",
                linewidth=1.6,
                label="Pareto Frontier",
                zorder=3,
            )

        max_cost = max(p[2] for p in points)
        ax.set_xlim(-max(0.01, max_cost * 0.08), max(0.1, max_cost * 1.18))
        ax.set_ylim(-3, 108)
        ax.set_xlabel("Cost per 100 Decisions (USD)")
        ax.set_ylabel("Optimal Accuracy (%)")
        ax.set_title("Quality vs. Cost Pareto Frontier (Track 1)")
        ax.legend(loc="lower right")
        png_path, svg_path = _save_fig(fig, output_dir, name)
    finally:
        plt.close(fig)

    return PlotArtifact(name=name, png_path=png_path, svg_path=svg_path, generated=True)


def _plot_dm_calibration_and_abstention(
    decisions: Sequence[DecisionLogRow],
    output_dir: Path,
) -> PlotArtifact:
    name = "dm_calibration_and_abstention"
    dm_rows = select_dm_decisions(decisions)
    if not dm_rows:
        return PlotArtifact(name=name, png_path=None, svg_path=None, generated=False, reason="No Decision Model decisions found for calibration")

    by_model: dict[str, list[DecisionLogRow]] = defaultdict(list)
    for row in dm_rows:
        by_model[row.model_key].append(row)

    model_keys = sorted(by_model.keys())
    fig, (ax_rel, ax_abs) = plt.subplots(1, 2, figsize=(12.0, 5.0))
    try:
        ax_rel.plot([0, 1], [0, 1], linestyle="--", color="#6B7280", linewidth=1.2, label="Perfect Calibration")

        for idx, mkey in enumerate(model_keys):
            cal = compute_dm_calibration(by_model[mkey])
            color = PALETTE[idx % len(PALETTE)]
            xs = [b.mean_confidence for b in cal.reliability_bins if b.count > 0 and b.mean_confidence is not None and b.empirical_accuracy is not None]
            ys = [b.empirical_accuracy for b in cal.reliability_bins if b.count > 0 and b.mean_confidence is not None and b.empirical_accuracy is not None]
            if xs:
                ax_rel.plot(
                    xs,
                    ys,
                    marker="o",
                    linewidth=2.0,
                    color=color,
                    label=f"{mkey} (ECE={cal.ece:.3f}, Brier={cal.brier_score:.3f})",
                )

            covs = [pt.coverage * 100.0 for pt in cal.abstention_curve if pt.accuracy is not None]
            accs = [pt.accuracy * 100.0 for pt in cal.abstention_curve if pt.accuracy is not None]
            if covs:
                ax_abs.plot(
                    covs,
                    accs,
                    marker="o",
                    markersize=3.5,
                    linewidth=2.0,
                    color=color,
                    label=f"{mkey} (opt_mass={cal.mean_optimal_mass:.2f})"
                    if cal.mean_optimal_mass is not None
                    else mkey,
                )

        ax_rel.set_xlim(0.0, 1.02)
        ax_rel.set_ylim(0.0, 1.05)
        ax_rel.set_xlabel("Mean Reported Confidence")
        ax_rel.set_ylabel("Empirical Optimal Accuracy")
        ax_rel.set_title("DM Top-1 Confidence Reliability Diagram")
        ax_rel.legend(loc="upper left", fontsize=8)

        ax_abs.set_xlim(0.0, 103.0)
        ax_abs.set_ylim(0.0, 105.0)
        ax_abs.set_xlabel("Coverage (% Retained Decisions)")
        ax_abs.set_ylabel("Retained Optimal Accuracy (%)")
        ax_abs.set_title("Selective Abstention Accuracy vs. Coverage")
        ax_abs.legend(loc="lower left", fontsize=8)

        fig.tight_layout()
        png_path, svg_path = _save_fig(fig, output_dir, name)
    finally:
        plt.close(fig)

    return PlotArtifact(name=name, png_path=png_path, svg_path=svg_path, generated=True)


def generate_all_plots(
    decisions: Sequence[DecisionLogRow],
    episodes: Sequence[EpisodeLogRow],
    output_dir: Path,
    *,
    n_bootstrap: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> list[PlotArtifact]:
    """Generate all 9 Milestone 1 charts (PNG + SVG), gracefully skipping absent experiments."""
    _apply_style()
    return [
        _plot_t2_core_accuracy_illegal(decisions, output_dir, n_bootstrap=n_bootstrap, seed=seed),
        _plot_t1_tactical_heatmap(decisions, output_dir),
        _plot_t3_rendering_robustness(decisions, output_dir, n_bootstrap=n_bootstrap, seed=seed),
        _plot_t4_legality_schema_delta(decisions, output_dir, n_bootstrap=n_bootstrap, seed=seed),
        _plot_t5_prompt_formats(decisions, output_dir, n_bootstrap=n_bootstrap, seed=seed),
        _plot_t6_match_outcomes(episodes, output_dir),
        _plot_latency_cdf(decisions, output_dir),
        _plot_pareto_quality_vs_cost(decisions, output_dir),
        _plot_dm_calibration_and_abstention(decisions, output_dir),
    ]
