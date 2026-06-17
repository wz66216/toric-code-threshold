from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, PathPatch
from matplotlib.path import Path as MplPath
import networkx as nx
from networkx.algorithms.matching import min_weight_matching

from .decoder import _defects_from_syndrome, decode_mwpm, toric_distance
from .lattice import ToricChain, Vertex, random_error
from .noisy import match_spacetime_events, sample_noisy_trial


def _segment_points(a: Vertex, b: Vertex, L: int):
    x1, y1 = a
    x2, y2 = b
    dx = (x2 - x1) % L
    dy = (y2 - y1) % L
    if dx == 1 and dy == 0:
        return [(x1, y1), (x2, y2)]
    if dx == 0 and dy == 1:
        return [(x1, y1), (x2, y2)]
    if dx == L - 1 and dy == 0:
        return [(0, y1), (L - 1, y1)]
    if dx == 0 and dy == L - 1:
        return [(x1, 0), (x1, L - 1)]
    return [(x1, y1), (x2, y2)]


def _toric_offset(delta: int, L: int) -> int:
    if delta > L / 2:
        return delta - L
    if delta < -L / 2:
        return delta + L
    return delta


def _quadratic_curve_points(a: Vertex, b: Vertex, L: int, bow_scale: float = 0.18):
    x1, y1 = a
    x2, y2 = b
    dx = _toric_offset(x2 - x1, L)
    dy = _toric_offset(y2 - y1, L)
    end = (x1 + dx, y1 + dy)
    mx, my = (x1 + end[0]) / 2, (y1 + end[1]) / 2
    ctrl = (mx - bow_scale * dy, my + bow_scale * dx)
    return [(x1, y1), ctrl, end]


def _draw_toric_background(ax, L):
    for x in range(L):
        for y in range(L):
            ax.scatter([x], [y], s=12, color="lightgray", zorder=1)
    for x in range(L):
        for y in range(L):
            for a, b in [((x, y), ((x + 1) % L, y)), ((x, y), (x, (y + 1) % L))]:
                dx = b[0] - a[0]
                dy = b[1] - a[1]
                if abs(dx) == 1 and abs(dy) == 0 or abs(dx) == 0 and abs(dy) == 1:
                    ax.plot([a[0], b[0]], [a[1], b[1]], color="lightgray", lw=0.7, zorder=0)
                else:
                    pts = _quadratic_curve_points(a, b, L, bow_scale=0.22)
                    path = MplPath(pts, [MplPath.MOVETO, MplPath.CURVE3, MplPath.CURVE3])
                    ax.add_patch(PathPatch(path, edgecolor="lightgray", facecolor="none", lw=0.7, zorder=0))


def _draw_chain_edges(ax, chain, L, color, linewidth=2.5, alpha=1.0, label=None, zorder=3):
    added_h = added_v = False
    for x in range(L):
        for y in range(L):
            if chain.horizontal[x, y]:
                a, b = (x, y), ((x + 1) % L, y)
                if x + 1 < L:
                    pts = [a, b]
                    ax.plot([p[0] for p in pts], [p[1] for p in pts], color=color, lw=linewidth, alpha=alpha, zorder=zorder, label=label if not added_h else None)
                else:
                    pts = _quadratic_curve_points(a, b, L)
                    path = MplPath(pts, [MplPath.MOVETO, MplPath.CURVE3, MplPath.CURVE3])
                    ax.add_patch(PathPatch(path, edgecolor=color, facecolor="none", lw=linewidth, alpha=alpha, zorder=zorder, label=label if not added_h else None))
                added_h = True
            if chain.vertical[x, y]:
                a, b = (x, y), (x, (y + 1) % L)
                if y + 1 < L:
                    pts = [a, b]
                    ax.plot([p[0] for p in pts], [p[1] for p in pts], color=color, lw=linewidth, alpha=alpha, zorder=zorder, label=label if not added_v else None)
                else:
                    pts = _quadratic_curve_points(a, b, L)
                    path = MplPath(pts, [MplPath.MOVETO, MplPath.CURVE3, MplPath.CURVE3])
                    ax.add_patch(PathPatch(path, edgecolor=color, facecolor="none", lw=linewidth, alpha=alpha, zorder=zorder, label=label if not added_v else None))
                added_v = True


def _draw_defects(ax, syndrome, L, zorder=4):
    defects = _defects_from_syndrome(syndrome)
    if defects:
        xs, ys = zip(*defects)
        ax.scatter(xs, ys, s=80, color="red", edgecolors="darkred", linewidths=1.5, zorder=zorder)


def _draw_matching(ax, syndrome, L, zorder=5):
    defects = _defects_from_syndrome(syndrome)
    if len(defects) < 2:
        return
    graph = nx.Graph()
    for i, d in enumerate(defects):
        graph.add_node(i, defect=d)
    for i in range(len(defects)):
        for j in range(i + 1, len(defects)):
            graph.add_edge(i, j, weight=toric_distance(defects[i], defects[j], L))
    matching = min_weight_matching(graph, weight="weight")
    for i, j in matching:
        a, b = defects[i], defects[j]
        pts = _quadratic_curve_points(a, b, L, bow_scale=0.15)
        path = MplPath(pts, [MplPath.MOVETO, MplPath.CURVE3, MplPath.CURVE3])
        ax.add_patch(PathPatch(path, edgecolor="green", facecolor="none", linestyle="--", lw=1.8, zorder=zorder))


def _draw_logical_cuts(ax, L, parity):
    ax.axhline(-0.5, color="orange", linestyle=":", lw=1.5)
    ax.axvline(-0.5, color="orange", linestyle=":", lw=1.5)
    success = not any(parity)
    text = f"{'SUCCESS' if success else 'LOGICAL FAILURE'}\nX: {'YES' if parity[0] else 'no'} | Y: {'YES' if parity[1] else 'no'}"
    ax.text(L / 2 - 0.5, L - 0.2, text, ha="center", va="top", fontsize=10, fontweight="bold", bbox=dict(boxstyle="round", facecolor="white", alpha=0.9), color=("green" if success else "red"))


def _legend_if_needed(ax, **kwargs):
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(**kwargs)


def _fig_ax(L):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-1, L)
    ax.set_ylim(-1.5, L)
    ax.axis("off")
    return fig, ax


def generate_trial_frames(L, p, seed=None, rng=None):
    if rng is None:
        import numpy as np
        rng = np.random.default_rng(seed)
    error = random_error(L, p, rng)
    syndrome = error.syndrome()
    recovery = decode_mwpm(syndrome)
    combined = error + recovery
    parity = combined.logical_parity()
    w = error.weight()
    n = len(_defects_from_syndrome(syndrome))
    frames = []
    fig, ax = _fig_ax(L); _draw_toric_background(ax, L); _draw_chain_edges(ax, error, L, "red", linewidth=3); ax.set_title(f"Error Chain (L={L}, p={p}, w={w})"); fig.tight_layout(); frames.append((fig, ax.get_title()))
    fig, ax = _fig_ax(L); _draw_toric_background(ax, L); _draw_chain_edges(ax, error, L, "red", linewidth=1.5, alpha=0.4); _draw_defects(ax, syndrome, L); ax.set_title(f"Syndrome ({n} defects)"); fig.tight_layout(); frames.append((fig, ax.get_title()))
    fig, ax = _fig_ax(L); _draw_toric_background(ax, L); _draw_defects(ax, syndrome, L); _draw_matching(ax, syndrome, L); ax.set_title("MWPM Pairing"); fig.tight_layout(); frames.append((fig, ax.get_title()))
    fig, ax = _fig_ax(L); _draw_toric_background(ax, L); _draw_chain_edges(ax, recovery, L, "blue", linewidth=3, label="Recovery"); _legend_if_needed(ax, loc="lower right"); ax.set_title(f"Recovery Chain (w={recovery.weight()})"); fig.tight_layout(); frames.append((fig, ax.get_title()))
    fig, ax = _fig_ax(L); _draw_toric_background(ax, L); _draw_chain_edges(ax, error, L, "red", linewidth=1.5, alpha=0.6, label="Error"); _draw_chain_edges(ax, recovery, L, "blue", linewidth=1.5, alpha=0.6, label="Recovery"); _draw_chain_edges(ax, error + recovery, L, "gray", linewidth=1.0, alpha=0.5); _draw_logical_cuts(ax, L, parity); _legend_if_needed(ax, loc="lower right"); ax.set_title(f"Combined Chain (w={w}) - {'SUCCESS' if not any(parity) else 'LOGICAL FAILURE'}"); fig.tight_layout(); frames.append((fig, ax.get_title()))
    return frames


def save_trial_frames(L, p, seed, output_dir, dpi=150):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    frames = generate_trial_frames(L, p, seed=seed)
    paths = []
    try:
        for i, (fig, _) in enumerate(frames, 1):
            path = out / f"frame_{i}.png"
            fig.savefig(path, dpi=dpi)
            paths.append(str(path))
    finally:
        for fig, _ in frames:
            plt.close(fig)
    return paths


def save_spacetime_detection_plot(
    L,
    T,
    p,
    q,
    seed,
    path,
    *,
    decoder_p=None,
    decoder_q=None,
    dpi=150,
):
    decoder_p = p if decoder_p is None else decoder_p
    decoder_q = q if decoder_q is None else decoder_q
    total_error, events = sample_noisy_trial(L=L, T=T, p=p, q=q, seed=seed)
    pairs = match_spacetime_events(events, L=L, p=decoder_p, q=decoder_q)

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    try:
        ax.set_title(
            f"Spacetime Detection Events (L={L}, T={T}, p={p}, q={q})\n"
            f"{len(events)} events, {'failure' if total_error.is_logical_failure() else 'no logical error before recovery'}"
        )
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("t")
        ax.set_xlim(-0.5, L - 0.5)
        ax.set_ylim(-0.5, L - 0.5)
        ax.set_zlim(-0.5, max(T, 1) - 0.5)
        ax.set_xticks(range(L))
        ax.set_yticks(range(L))
        ax.set_zticks(range(T + 1))
        ax.view_init(elev=22, azim=-55)

        for t in range(T + 1):
            alpha = 0.08 if t % 2 else 0.14
            xs = [0, L - 1, L - 1, 0, 0]
            ys = [0, 0, L - 1, L - 1, 0]
            zs = [t] * 5
            ax.plot(xs, ys, zs, color="gray", lw=0.8, alpha=alpha)

        if events:
            ax.scatter(
                [event.x for event in events],
                [event.y for event in events],
                [event.t for event in events],
                s=56,
                color="crimson",
                edgecolors="darkred",
                depthshade=False,
                label="detection event",
            )
        else:
            ax.scatter([], [], [], s=56, color="crimson", label="detection event")

        for index, (a, b) in enumerate(pairs):
            ax.plot(
                [a.x, b.x],
                [a.y, b.y],
                [a.t, b.t],
                color="royalblue",
                lw=1.8,
                alpha=0.8,
                label="MWPM pair" if index == 0 else None,
            )

        if pairs or events:
            ax.legend(loc="upper left")
        fig.tight_layout()
        fig.savefig(output, dpi=dpi)
    finally:
        plt.close(fig)
    return str(output)
