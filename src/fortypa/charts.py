"""Static figures per PA: the 2017 / 2025 class maps side by side, and the line chart of forest class areas."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

from .gee import CLASSES, FOREST  # noqa: E402

INK, MUTED, GRID = "#1F1F1F", "#6B6B6B", "#E6E6E3"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})


def fmt_ha(v, d=0):
    """Vietnamese number format: 12.345,6"""
    s = f"{v:,.{d}f}"
    return s.replace(",", "_").replace(".", ",").replace("_", ".")


def _title(pa):
    return f"{pa.NAME} – {pa.DESIG_ENG}"


def map_figure(pa, thumbs, years, out_path):
    """Two class maps (first and last year) with a shared legend underneath. thumbs: {year: png path}."""
    imgs = {y: plt.imread(thumbs[y]) for y in years if thumbs.get(y) and Path(thumbs[y]).exists()}
    aspect = max([im.shape[0] / im.shape[1] for im in imgs.values()] or [0.75])  # height / width of the maps
    map_w, top_in, bottom_in = 5.5, 0.9, 0.9  # inches: each map's width, title and legend bands
    fig_h = map_w * aspect + top_in + bottom_in
    fig, axes = plt.subplots(1, len(years), figsize=(map_w * len(years), fig_h))
    for ax, y in zip(axes, years):
        ax.set_axis_off()
        if y in imgs:
            ax.imshow(imgs[y])
        else:
            ax.text(0.5, 0.5, "Không tải được ảnh", ha="center", va="center", color=MUTED, transform=ax.transAxes)
        ax.set_title(str(y), fontsize=13, fontweight="bold")
    handles = [Patch(facecolor=c.color, edgecolor="#BDBDBD", label=c.label) for c in CLASSES.itertuples()]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=9)
    fig.suptitle(_title(pa), fontsize=13)
    fig.subplots_adjust(left=0.01, right=0.99, top=1 - top_in / fig_h, bottom=bottom_in / fig_h, wspace=0.03)
    fig.savefig(out_path, dpi=110, facecolor="white")
    plt.close(fig)
    return out_path


def change_figure(pa, change_png, trans_pa, years, out_path):
    """Change map with a legend giving each change category's area (ha, % of the analysed area) at 10 m."""
    from .gee import CHANGES
    img = plt.imread(change_png)
    aspect = img.shape[0] / img.shape[1]
    map_w = 6.5
    fig = plt.figure(figsize=(map_w + 5.2, max(map_w * aspect, 4.2) + 0.8))
    ax = fig.add_axes((0.005, 0.02, map_w / (map_w + 5.2), 0.84))
    ax.imshow(img)
    ax.set_axis_off()
    by = trans_pa.groupby("change").area_ha.sum().reindex(CHANGES.code, fill_value=0)
    total = by.sum()
    handles = [Patch(facecolor=c.color, edgecolor="#BDBDBD",
                     label=f"{c.label}\n{fmt_ha(by[c.code])} ha ({fmt_ha(100 * by[c.code] / total, 1)}%)")
               for c in CHANGES.itertuples()][::-1]  # most consequential change first
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(map_w / (map_w + 5.2) + 0.01, 0.86),
               frameon=False, fontsize=9, labelspacing=0.9, handleheight=1.6)
    changed = total - by[0]
    fig.text(map_w / (map_w + 5.2) + 0.015, 0.04,
             f"Đổi lớp: {fmt_ha(changed)} ha ({fmt_ha(100 * changed / total, 1)}% diện tích phân tích).\n"
             "Diện tích tính ở 10 m. Trên bản đồ, mỗi điểm ảnh gộp nhiều\n"
             "pixel 10 m: màu = loại thay đổi chiếm ưu thế, độ đậm =\n"
             "tỷ lệ pixel 10 m đổi lớp.", fontsize=8.5, color=MUTED, va="bottom")
    fig.suptitle(f"Thay đổi loại rừng {years[0]} → {years[-1]} · {_title(pa)}", fontsize=12, x=0.01, ha="left")
    fig.savefig(out_path, dpi=110, facecolor="white")
    plt.close(fig)
    return out_path


def area_table(areas_pa, years):
    """year × forest class (ha), columns in FOREST order; classes absent from the PA are 0."""
    t = (areas_pa[areas_pa.value != 6].pivot_table(index="year", columns="value", values="area_ha", aggfunc="sum")
         .reindex(index=list(years), columns=list(FOREST.value)).fillna(0))
    return t


def _style(ax, years, ylabel):
    ax.set_xlim(years[0] - 0.3, years[-1] + 0.3)
    ax.set_xticks(list(years))
    ax.tick_params(axis="x", labelsize=8)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: fmt_ha(v)))
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)


def area_chart(pa, areas_pa, years, out_path):
    """Line chart per forest class: area (ha) on the left, change since the first year (ha) on the right.

    Classes differ by orders of magnitude (hundreds of thousands of ha of primary forest next to a few ha of tree
    crops), so on the area panel small classes sit flat on the baseline; the change panel puts every class on the
    same footing, in the same unit, without a second y-axis.
    """
    t = area_table(areas_pa, years)
    keep = [v for v in t.columns if t[v].max() > 0]
    meta = FOREST.set_index("value")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))
    for v in keep:
        kw = dict(color=meta.color[v], linewidth=2, marker="o", markersize=4)
        ax1.plot(t.index, t[v], label=meta.label[v], **kw)
        ax2.plot(t.index, t[v] - t[v].iloc[0], **kw)
    ax1.set_ylim(bottom=0)
    ax2.axhline(0, color=MUTED, linewidth=0.8)
    _style(ax1, years, "Diện tích (ha)")
    _style(ax2, years, "ha")
    ax1.set_title("Diện tích theo loại rừng", fontsize=11, loc="left")
    ax2.set_title(f"Thay đổi so với năm {years[0]}", fontsize=11, loc="left")
    fig.suptitle(f"{_title(pa)} · {years[0]}–{years[-1]}", fontsize=12, x=0.01, ha="left")
    # legend ordered by area in the last year, largest first, so it matches the vertical order of the lines
    order = sorted(keep, key=lambda v: -t[v].iloc[-1])
    h = {lab: hd for hd, lab in zip(*ax1.get_legend_handles_labels())}
    fig.legend([h[meta.label[v]] for v in order], [meta.label[v] for v in order], loc="lower center",
               ncol=min(len(order), 3), frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.12 if len(order) > 3 else 0.07, 1, 0.95))
    fig.savefig(out_path, dpi=110, facecolor="white")
    plt.close(fig)
    return out_path
