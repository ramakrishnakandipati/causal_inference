import sys, time, traceback
sys.path.insert(0, ".")
import numpy as np
from src.common import FEATURES, load_config
from src.features import load_data, split_data
from src.models import ESTIMATORS, device_info
from src.metrics import pehe, auuc
cfg = load_config()
cfg["training"]["nn"]["epochs"] = 8
cfg["training"]["forest"]["n_estimators"] = 48
cfg["training"]["uplift_rf"]["n_estimators"] = 20
print(device_info(cfg))
sp = split_data(load_data(cfg), cfg)
tr, te = sp["train"].iloc[:8000], sp["test"].iloc[:3000]
for name, cls in ESTIMATORS.items():
    t0 = time.time()
    try:
        m = cls(cfg)
        m.fit_with_ci(tr[FEATURES], tr["T"].values, tr["Y_margin"].values, {"Y_conv": tr["Y_conv"].values}, B=2)
        tau = m.predict(te[FEATURES]); ci = m.predict_interval(te[FEATURES])
        print(f"{name:16s} ok {time.time()-t0:5.1f}s  pehe={pehe(tau, te.tau_margin_true):7.2f}  corr={np.corrcoef(tau, te.tau_margin_true)[0,1]:.3f}  ci={'yes' if ci else 'no'} width={(ci[1]-ci[0]).mean() if ci else float('nan'):.2f}")
    except Exception as e:
        print(name, "FAIL", repr(e)); traceback.print_exc()
