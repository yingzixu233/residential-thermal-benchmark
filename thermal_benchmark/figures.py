"""Generate exportable figures from recorded results, without retraining."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .models import MODEL_NAMES

COLORS = dict(zip(MODEL_NAMES, ["#7f8c8d", "#3969ac", "#0b806b", "#9b59b6", "#d98c10", "#ce4b40"]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", default="results")
    args = p.parse_args()
    root = Path(args.results)
    out = root/"figures"
    out.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "savefig.dpi": 220})
    m = pd.read_csv(root/"metrics.csv")
    h = pd.read_csv(root/"by_horizon.csv")
    aggregate = m.groupby(["days", "noise_C", "model", "mode"])[["mae_C", "rmse_C"]].agg(["mean", "std"])
    aggregate.columns = ["_".join(c) for c in aggregate.columns]
    aggregate.to_csv(root/"summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(9.3, 3.0), layout="constrained")
    for model in MODEL_NAMES:
        a = m.query("noise_C == 0 and mode == 'rollout_path' and model == @model")
        g = a.groupby("days").rmse_C.agg(["mean", "std"])
        axes[0].errorbar(g.index, g["mean"], yerr=g["std"], marker="o", ms=4,
                         capsize=2, color=COLORS[model], label=model)
        a = h.query("noise_C == 0 and days == 10 and model == @model")
        g = a.groupby("horizon_h").rmse_C.agg(["mean", "std"])
        axes[1].plot(g.index, g["mean"], color=COLORS[model], label=model)
        axes[1].fill_between(g.index, g["mean"]-g["std"], g["mean"]+g["std"],
                              color=COLORS[model], alpha=.10)
    axes[0].set(title="A  Recent training budget", xlabel="Training observations (days)",
                ylabel="24 h path RMSE (C)", xticks=[3, 5, 10])
    axes[1].set(title="B  Ten-day training: recursive forecast", xlabel="Forecast horizon (hours)",
                ylabel="RMSE at horizon (C)", xticks=[1, 6, 12, 18, 24])
    for ax in axes: ax.grid(alpha=.18)
    axes[1].legend(fontsize=7, ncol=2)
    fig.savefig(out/"accuracy.png"); fig.savefig(out/"accuracy.pdf"); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(9.3, 3.0), layout="constrained")
    for ax, mode, title in zip(axes, ["one_step", "rollout_path"], ["One-step predictions", "24 h recursive path"]):
        for model in MODEL_NAMES:
            a = m.query("days == 10 and mode == @mode and model == @model")
            g = a.groupby("noise_C").rmse_C.agg(["mean", "std"])
            ax.errorbar(g.index, g["mean"], yerr=g["std"], marker="o", ms=4,
                         capsize=2, color=COLORS[model], label=model)
        ax.set(title=title, xlabel="Input temperature noise SD (C)", ylabel="RMSE (C)", xticks=[0, .2, .5])
        ax.grid(alpha=.18)
    axes[1].legend(fontsize=7, ncol=2)
    fig.savefig(out/"noise_robustness.png"); plt.close(fig)
    importance = pd.read_csv(root/"shap"/"importance.csv")
    fig, axes = plt.subplots(1, 2, figsize=(9.3, 3.1), layout="constrained")
    for ax, model in zip(axes, ["RF", "XGBoost"]):
        q = importance.query("zone == 'ALL' and model == @model").nlargest(8, "mean_abs_shap_C").sort_values("mean_abs_shap_C")
        ax.barh(q.feature, q.mean_abs_shap_C, color=COLORS[model], height=.7)
        ax.set(title=f"{model}: top 8 features", xlabel="Mean absolute SHAP value (C)")
        ax.grid(axis="x", alpha=.18)
    fig.savefig(out/"shap_global.png"); fig.savefig(out/"shap_global.pdf"); plt.close(fig)
    # One fixed, pre-specified origin (the first test timestamp); never cherry-picked.
    fig, axes = plt.subplots(1, 3, figsize=(10, 3), layout="constrained")
    for ax, zone in zip(axes, [0, 4, 8]):
        for model in MODEL_NAMES:
            a = np.load(root/"predictions"/f"{model}_10d_s0_n0.npz")
            ax.plot(np.arange(1,25), a["rollout"][0,:,zone], color=COLORS[model], label=model, lw=1)
        ax.plot(np.arange(1,25), a["rollout_truth"][0,:,zone], color="black", lw=1.8, label="Measured")
        ax.set(title=f"Zone {zone+1:02d}", xlabel="Hours after first test origin", ylabel="Temperature (C)")
    axes[-1].legend(fontsize=6, ncol=2)
    fig.savefig(out/"example_rollout.png"); plt.close(fig)


if __name__ == "__main__":
    main()
