"""Bar chart of test-year recall: random vs past-breaks rule vs model, from the logged results.

Usage:
    python -m src.experiments.plot_results      # results/test_recall.png
"""
import matplotlib.pyplot as plt
import pandas as pd

from src.model import ROOT

r = pd.read_csv(ROOT / "results" / "annual_design_validation_and_test.csv").set_index("run")
rule, model = r.loc["annual-wall-rule-nearby-test"], r.loc["annual-wall-lgbm-nearby-test"]
groups = {"All pipes\nrecall at 0.25% of length / yr": (0.0025, rule["pooled_R@0.25%"], model["pooled_R@0.25%"]),
          "Pipes with no break history\nrecall at 1% of length": (0.01, rule["R@1%_never_broken"], model["R@1%_never_broken"])}

fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
for ax, (title, values) in zip(axes, groups.items()):
    bars = ax.bar(["random", "rule: past\nbreaks per km", "model"], [v * 100 for v in values],
                  color=["#c9c8c2", "#52514e", "#2a78d6"], width=0.6)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v * 100, (f"{v * 100:.2f}%" if v < 0.005 else f"{v * 100:.1f}%"), ha="center", va="bottom", fontsize=9)
    ax.set_title(title, fontsize=10, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylabel("share of breaks caught (%)")
fig.suptitle("Test years 2021–2025 (855 breaks): breaks caught by each replacement list", fontsize=10, x=0.01, ha="left")
fig.tight_layout()
fig.savefig(ROOT / "results" / "test_recall.png", dpi=170)
