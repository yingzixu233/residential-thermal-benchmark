"""python -m thermal_benchmark.run --output results"""
import argparse
import json
import platform
import time
import importlib.metadata
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from .data import load_data, split_data, describe_data, VALID_END, TEMPERATURES
from .models import MODEL_NAMES, select_model
from .evaluation import metrics, temperature_noise, recursive_rollout


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="data/house_data.csv")
    p.add_argument("--output", default="results")
    p.add_argument("--days", nargs="+", type=int, default=[3, 5, 10])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--noise", nargs="+", type=float, default=[0., 0.2, 0.5])
    p.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=MODEL_NAMES)
    p.add_argument("--epochs", type=int, default=1500)
    args = p.parse_args()
    out = Path(args.output)
    for folder in (out, out/"models", out/"predictions", out/"training"):
        folder.mkdir(parents=True, exist_ok=True)
    df = load_data(args.data)
    manifest = {"arguments": vars(args), "data": describe_data(df, args.data),
                "python": platform.python_version(), "platform": platform.platform(),
                "versions": {k: importlib.metadata.version(k) for k in
                             ["numpy", "pandas", "torch", "scikit-learn", "xgboost", "shap"]},
                "selection": "clean validation one-step RMSE, train-only scaling; no refit",
                "noise": "Gaussian indoor-temperature inputs only; uncorrupted recorded targets; rollout initialization only",
                "forcing": "recorded future heating/weather (conditional forecast)",
                "horizon": 24, "seed_sd": "sample SD across 3 seeds, not a confidence interval"}
    (out/"manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    summary, zones, horizons, selection = [], [], [], []
    started = time.perf_counter()
    for days in args.days:
        train, valid, test = split_data(df, days)
        for seed in args.seeds:
            for name in args.models:
                key = {"days": days, "seed": seed, "model": name}
                begin = time.perf_counter()
                model, val_score, params, trials = select_model(name, seed, train, valid, args.epochs)
                elapsed = time.perf_counter()-begin
                selection.append({**key, "validation_rmse_C": val_score,
                                  "search_seconds": elapsed, "selected_parameters": json.dumps(params)})
                (out/"training"/f"{name}_{days}d_s{seed}.json").write_text(
                    json.dumps(trials, indent=2), encoding="utf-8")
                if days == 10 and seed == 0:
                    joblib.dump(model, out/"models"/f"{name}.joblib", compress=3)
                for sigma in args.noise:
                    context = {**key, "noise_C": sigma}
                    one = model.predict(temperature_noise(test[0], sigma, seed))
                    roll, truth = recursive_rollout(model, df.iloc[VALID_END:], 24, sigma, seed)
                    for mode, pred, target in [("one_step", one, test[1]),
                                               ("rollout_path", roll, truth),
                                               ("rollout_h24", roll[:, -1], truth[:, -1])]:
                        summary.append({**context, "mode": mode, **metrics(pred, target)})
                        for j, zone in enumerate(TEMPERATURES):
                            zones.append({**context, "mode": mode, "zone": zone,
                                          **metrics(pred[..., j], target[..., j])})
                    for h in range(24):
                        horizons.append({**context, "horizon_h": h+1,
                                         **metrics(roll[:, h], truth[:, h])})
                    np.savez_compressed(out/"predictions"/f"{name}_{days}d_s{seed}_n{sigma:g}.npz",
                                        one_step=one.astype("float32"), one_truth=test[1].astype("float32"),
                                        rollout=roll.astype("float32"), rollout_truth=truth.astype("float32"))
                print(f"{days:2d}d seed={seed} {name:12s} val={val_score:.4f} search={elapsed:.1f}s", flush=True)
                # Incremental outputs also make interrupted runs inspectable.
                for filename, rows in [("metrics", summary), ("by_zone", zones),
                                       ("by_horizon", horizons), ("selection", selection)]:
                    pd.DataFrame(rows).to_csv(out/f"{filename}.csv", index=False)
    manifest["wall_seconds"] = time.perf_counter()-started
    manifest["complete"] = True
    (out/"manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Completed in {manifest['wall_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
