"""Build the prediction tasks from the logged tables, plus the gradient-boosting baseline."""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupShuffleSplit

HP = ["data", "noise", "H", "act", "opt", "lr", "batch_size", "init_scale", "wd"]


def build_task(runs, steps, task, k=10):
    """Returns (features DataFrame, text list, target, groups) with one row per run.

    T1: hyperparameters                          -> final val loss
    T2: hyperparameters + weights & loss at step k -> final val loss  (early prediction)
    Target is log10(final val loss) so tiny and huge losses are on a comparable scale.
    """
    df = runs.reset_index(drop=True)
    X = df[HP].copy()
    texts = [" ".join(f"{c}={r[c]:.3g}" if isinstance(r[c], float) else f"{c}={r[c]}" for c in HP)
             for _, r in df.iterrows()]
    if task == "T2":
        at_k = steps[steps["step"] == k].set_index("run_id").loc[df["run_id"]]
        w_cols = [c for c in at_k.columns if c.startswith("before_") and at_k[c].notna().any()]
        X = pd.concat([X, at_k[w_cols + ["val_loss", "val_acc"]].reset_index(drop=True)], axis=1)
        for i, (_, r) in enumerate(at_k.iterrows()):   # same info, as text for the SLM
            w = ",".join(f"{r[c]:.2f}" for c in w_cols if not np.isnan(r[c]))
            texts[i] += f" | step{k} val_loss={r['val_loss']:.3f} val_acc={r['val_acc']:.2f} w=[{w}]"
    y = np.log10(df["final_val_loss"].values)
    return X, texts, y, df["config_id"].values


def split(groups, seed=0):
    """Train/test split by config, so seeds of the same config never leak across the split."""
    return next(GroupShuffleSplit(1, test_size=0.25, random_state=seed).split(groups, groups=groups))


def score(y_true, y_pred):
    """MAE on log10 loss, and Spearman rank correlation (what matters for picking a config)."""
    return {"MAE": float(np.mean(np.abs(y_true - y_pred))), "Spearman": float(spearmanr(y_true, y_pred)[0])}


def gbm_baseline(X, y, tr, te):
    X = pd.get_dummies(X, columns=["data", "act", "opt"]).astype(float)   # NaN (missing H=3 weights) is OK
    model = HistGradientBoostingRegressor(max_iter=300).fit(X.iloc[tr], y[tr])
    return model.predict(X.iloc[te])
