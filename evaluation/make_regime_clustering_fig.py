#!/usr/bin/env python3
"""Unsupervised justification of the Group A / Group B split of Table 1, over
two features per log: log10 median inter-event time and log10 activity count.

Group A is high-frequency and small-vocabulary, Group B day-scale or
large-vocabulary; BPIC15-1 is flagged an outlier (shortest gaps, 398
activities) and assigned to Group B by the vocabulary criterion.

Outputs: paper_results/fig_regime_clustering.{pdf,png}
"""
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

DAY = 86400.0
# name, median dt (s), #Act, hand label; Table 1, recomputed by
# `python prepare_uq4ppm.py --stats`
#
# #Act counts distinct concept:name. UQ4PPM and the LA-CR paper count
# concept:name x lifecycle:transition, i.e. BPIC12 36 and BPIC13I 13 instead of
# 24 and 4; the other eight logs have a constant lifecycle. Substituting those
# two leaves the k=2 partition and the k=3 blocks below unchanged, and moves the
# silhouette at k=2 to 0.436 (excl. outlier) and 0.608 (all ten).
logs = [
    ("HelpDesk",   3.28*DAY,  14,  "B"),
    ("Sepsis",     6.5*60,    16,  "A"),
    ("BPIC13I",    10.1*60,   4,   "A"),
    ("BPIC20DD",   1.02*DAY,  17,  "B"),
    ("BPIC20RFP",  1.11*DAY,  19,  "B"),
    ("BPIC20PTC",  22.5*3600, 29,  "B"),
    ("BPIC20ID",   2.00*DAY,  34,  "B"),
    ("BPIC20TPD",  1.74*DAY,  51,  "B"),
    ("BPIC12",     48.0,      24,  "A"),
    ("BPIC15-1",   1.0,       398, "B"),   # outlier -> B by vocabulary
]
names = [r[0] for r in logs]
X = np.array([[np.log10(r[1]), np.log10(r[2])] for r in logs])
hand = np.array([r[3] for r in logs])
OUT = "BPIC15-1"
is_out = np.array([n == OUT for n in names])

cA, cB = "#1b9e77", "#7570b3"
col = np.where(hand == "A", cA, cB)

# silhouette over all 10 logs (argmax degenerates to peeling the outlier) and
# over the 9 without it
ks = list(range(2, 6))
Xz = StandardScaler().fit_transform(X)
sil_all = [silhouette_score(Xz, KMeans(k, n_init=25, random_state=0).fit_predict(Xz))
           for k in ks]
keep = ~is_out
X9z = StandardScaler().fit_transform(X[keep])
sil_9 = [silhouette_score(X9z, KMeans(k, n_init=25, random_state=0).fit_predict(X9z))
         for k in ks]
lab9 = KMeans(2, n_init=25, random_state=0).fit_predict(X9z)
print("k=2 on the 9 non-outlier logs vs hand label:")
for n, l in zip([m for m, o in zip(names, is_out) if not o], lab9):
    print(f"   {n:11s} cluster={l}")

plt.rcParams.update({"font.size": 13})
fig, ax = plt.subplots(1, 2, figsize=(11, 5.0))

# (a) Ward dendrogram with the k=2 and k=3 cuts
Z = linkage(Xz, method="ward")
dendrogram(Z, labels=names, ax=ax[0], color_threshold=Z[-2, 2],
           leaf_rotation=90, leaf_font_size=12)
d_k2 = 0.5 * (Z[-1, 2] + Z[-2, 2])   # cut here -> 2 clusters (isolates outlier)
d_k3 = 0.5 * (Z[-2, 2] + Z[-3, 2])   # cut here -> 3 clusters (Group A splits off)
ax[0].axhline(d_k2, ls=":",  c="crimson", lw=1.8)
ax[0].axhline(d_k3, ls="--", c="0.35",    lw=1.6)
trans = ax[0].get_yaxis_transform()
ax[0].text(0.015, d_k2, "$K{=}2$", transform=trans, color="crimson",
           va="bottom", ha="left", fontsize=13)
ax[0].text(0.015, d_k3, "$K{=}3$", transform=trans, color="0.25",
           va="bottom", ha="left", fontsize=13)
ax[0].set_ylabel("merge distance", fontsize=14)
ax[0].tick_params(axis="y", labelsize=12)

# (b) silhouette, all logs vs outlier excluded
ax[1].plot(ks, sil_all, "o--", c="#999999", lw=2.2, ms=9, label="all 10 logs")
ax[1].plot(ks, sil_9,   "o-",  c="#1b9e77", lw=2.2, ms=9, label="excl. outlier")
ax[1].scatter([2], [sil_9[0]], s=260, facecolor="none", edgecolor="crimson",
              linewidth=2.4, zorder=4)
ax[1].annotate("adopted $K{=}2$", (2, sil_9[0]), xytext=(2.25, sil_9[0]-0.05),
               fontsize=13, color="crimson")
ax[1].set_xticks(ks)
ax[1].set_xlabel("number of clusters $K$", fontsize=14)
ax[1].set_ylabel("mean silhouette", fontsize=14)
ax[1].tick_params(labelsize=12)
ax[1].legend(fontsize=12, loc="upper right")

fig.tight_layout()
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "paper_results")
fig.savefig(os.path.join(OUT_DIR, "fig_regime_clustering.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(OUT_DIR, "fig_regime_clustering.png"), dpi=150, bbox_inches="tight")
print("wrote fig_regime_clustering.{pdf,png}")
