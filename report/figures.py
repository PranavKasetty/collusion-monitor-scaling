"""Figures for the Swarm sprint report; numbers copied from RESULTS.md (GLM-5.3-flash, whole interleaved, 50 scenarios)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

N = [4, 8, 16]
C = {"stance": "#555555", "v2": "#D55E00", "v4": "#0072B2", "spread": "#009E73", "solo": "#CC79A7"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6})


def line(ax, ys, lo, hi, key, label, ls="-"):
    ax.fill_between(N, lo, hi, color=C[key], alpha=0.15, lw=0)
    ax.plot(N, ys, ls, color=C[key], lw=2, marker="o", ms=4, label=label)


fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.7))
ax = axs[0]
line(ax, [.95, .94, .94], [.90, .88, .90], [.99, .99, .98], "stance", "stance (count target votes)")
line(ax, [.97, .97, .91], [.94, .94, .85], [1.0, .99, .96], "spread", "v2, spread of agent scores", "--")
line(ax, [.69, .67, .76], [.58, .54, .66], [.79, .79, .86], "v4", "v4, coalition answer")
line(ax, [.23, .41, .44], [.14, .31, .33], [.33, .51, .54], "v2", "v2, coalition answer")
ax.set_title("(a) Detection: collusion vs control", fontsize=9)
ax.set_ylabel("AUROC"); ax.legend(fontsize=7, frameon=False, loc="lower right")
ax = axs[1]
line(ax, [.95, .94, .94], [.91, .90, .90], [.99, .98, .97], "stance", "stance")
line(ax, [.87, .68, .69], [.77, .54, .56], [.95, .80, .82], "v4", "v4")
line(ax, [.81, .52, .36], [.70, .38, .23], [.91, .66, .49], "v2", "v2")
ax.set_title("(b) Attribution: colluders vs other agents", fontsize=9)
ax.legend(fontsize=7, frameon=False, loc="lower left")
for ax in axs:
    ax.axhline(0.5, color="#999999", lw=0.8, ls=":")
    ax.set_xscale("log", base=2); ax.set_xticks(N); ax.set_xticklabels(N); ax.set_xlabel("committee size N")
    ax.set_ylim(0, 1.02)
fig.tight_layout(); fig.savefig("fig1.png", dpi=220)

fig, ax = plt.subplots(figsize=(7.2, 2.4))
groups = [("collusion vs control", [.69, .67, .76], [.58, .54, .66], [.79, .79, .86], "v4"),
          ("lone advocates vs control", [.67, .70, .73], [.56, .58, .63], [.76, .79, .83], "solo"),
          ("collusion vs lone advocates", [.54, .44, .55], [.46, .33, .45], [.62, .55, .64], "stance")]
w = 0.26
for i, (lab, ys, lo, hi, key) in enumerate(groups):
    xs = [j + (i - 1) * w for j in range(3)]
    err = None if lo is None else [[y - l for y, l in zip(ys, lo)], [h - y for y, h in zip(ys, hi)]]
    ax.bar(xs, ys, w * 0.9, color=C[key], label=lab, yerr=err, error_kw={"lw": 0.8, "capsize": 2})
ax.axhline(0.5, color="#999999", lw=0.8, ls=":")
ax.set_xticks(range(3)); ax.set_xticklabels([f"N = {n}" for n in N]); ax.set_ylim(0, 1); ax.set_ylabel("AUROC (v4 coalition)")
ax.legend(fontsize=7, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.18))
fig.tight_layout(); fig.savefig("fig2.png", dpi=220)
