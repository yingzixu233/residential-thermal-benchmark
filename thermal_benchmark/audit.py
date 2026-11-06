"""Audit saved predictions against every reported metric."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .evaluation import metrics


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results",default="results")
    args=p.parse_args()
    root=Path(args.results)
    m=pd.read_csv(root/"metrics.csv")
    z=pd.read_csv(root/"by_zone.csv")
    h=pd.read_csv(root/"by_horizon.csv")
    assert len(m)==486 and len(z)==4374 and len(h)==3888
    assert len(pd.read_csv(root/"selection.csv"))==54
    assert not m.duplicated(["days","seed","model","noise_C","mode"]).any()
    max_error=0.
    for key, group in m.groupby(["days","seed","model","noise_C"]):
        days,seed,model,noise=key
        a=np.load(root/"predictions"/f"{model}_{days}d_s{seed}_n{noise:g}.npz")
        assert a["one_step"].shape==(96,9) and a["rollout"].shape==(73,24,9)
        for mode,pred,target in [("one_step",a["one_step"],a["one_truth"]),
                                 ("rollout_path",a["rollout"],a["rollout_truth"]),
                                 ("rollout_h24",a["rollout"][:,-1],a["rollout_truth"][:,-1])]:
            row=group[group["mode"]==mode].iloc[0]
            for metric,value in metrics(pred,target).items():
                max_error=max(max_error,abs(value-row[metric]))
    assert max_error<1e-5
    additivity=json.loads((root/"shap/audit.json").read_text())
    assert all(a["maximum_additivity_error_C"]<1e-4 for a in additivity.values())
    report={"passed":True,"settings":162,"metric_rows":len(m),"zone_rows":len(z),
            "horizon_rows":len(h),"prediction_metric_max_error_C":max_error,
            "shap_additivity_passed":True}
    (root/"audit.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
