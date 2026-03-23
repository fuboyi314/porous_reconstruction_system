from __future__ import annotations

from matplotlib.figure import Figure

from app.core.dto import ComputedMetrics


def plot_pore_histogram(metrics: ComputedMetrics) -> Figure:
    figure = Figure(figsize=(5, 3), tight_layout=True)
    ax = figure.add_subplot(111)
    bin_edges = metrics.pore_size_bin_edges
    hist = metrics.pore_size_histogram
    if len(bin_edges) > 1 and hist:
        centers = [(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(len(bin_edges) - 1)]
        widths = [bin_edges[i + 1] - bin_edges[i] for i in range(len(bin_edges) - 1)]
        ax.bar(centers, hist, width=widths, color="#3A7AFE", alpha=0.85)
    ax.set_title("孔径分布")
    ax.set_xlabel("等效孔半径")
    ax.set_ylabel("频率")
    return figure
