"""TreeSHAP for ten-day, seed-zero models; train-only background."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import shap
from .data import load_data, split_data, FEATURES, TEMPERATURES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results")
    parser.add_argument("--data", default="data/house_data.csv")
    args = parser.parse_args()
    root = Path(args.results)
    out = root/"shap"
    out.mkdir(exist_ok=True)
    train, _, test = split_data(load_data(args.data), 10)
    background = train[0][np.linspace(0, len(train[0])-1, 100, dtype=int)]
    x = test[0]
    rows, audit = [], {}
    for name in ("RF", "XGBoost"):
        model = joblib.load(root/"models"/f"{name}.joblib")
        if name == "RF":
            explainer = shap.TreeExplainer(model, data=background,
                feature_perturbation="interventional", model_output="raw")
            values = np.asarray(explainer.shap_values(x, check_additivity=True))
            bases = np.asarray(explainer.expected_value)
        else:
            values_by_zone, bases_by_zone = [], []
            for estimator in model.estimators_:
                explainer = shap.TreeExplainer(estimator, data=background,
                    feature_perturbation="interventional", model_output="raw")
                values_by_zone.append(explainer.shap_values(x, check_additivity=True))
                bases_by_zone.append(float(explainer.expected_value))
            values = np.stack(values_by_zone, axis=-1)
            bases = np.asarray(bases_by_zone)
        if values.shape != (96, 20, 9):
            raise AssertionError(f"Unexpected SHAP shape {values.shape}")
        reconstructed = values.sum(axis=1)+bases
        discrepancy = float(np.max(np.abs(reconstructed-model.predict(x))))
        if discrepancy > 1e-4:
            raise AssertionError(f"SHAP reconstruction error {discrepancy}")
        np.savez_compressed(out/f"{name}_values.npz", shap_values=values,
                            expected_values=bases, inputs=x, background=background,
                            feature_names=FEATURES, zone_names=TEMPERATURES)
        for j, feature in enumerate(FEATURES):
            rows.append({"model": name, "zone": "ALL", "feature": feature,
                         "mean_abs_shap_C": float(np.abs(values[:, j, :]).mean())})
            for k, zone in enumerate(TEMPERATURES):
                rows.append({"model": name, "zone": zone, "feature": feature,
                             "mean_abs_shap_C": float(np.abs(values[:, j, k]).mean())})
        audit[name] = {"maximum_additivity_error_C": discrepancy,
                       "explained_samples": len(x), "outputs": 9,
                       "background_rows": len(background), "model_seed": 0,
                       "interpretation": "one-step model attribution, not causation or rollout attribution"}
        print(name, audit[name], flush=True)
    pd.DataFrame(rows).to_csv(out/"importance.csv", index=False)
    (out/"audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
