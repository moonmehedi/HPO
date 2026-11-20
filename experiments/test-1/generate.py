"""Sample hyperparameter configs, train the tiny MLP on each, and save the logs as 3 tables."""
import numpy as np
import pandas as pd
from tiny_mlp import train

OPTS, ACT_LIST, DATA_LIST = ["sgd", "momentum", "adam"], ["tanh", "relu"], ["moons", "circles", "xor"]


def sample_config(rng):
    """One random config (log-uniform for scale-type hyperparameters)."""
    return {"data": rng.choice(DATA_LIST), "noise": float(rng.choice([0.1, 0.2, 0.3])),
            "H": int(rng.choice([2, 3])), "act": rng.choice(ACT_LIST), "opt": rng.choice(OPTS),
            "lr": float(10 ** rng.uniform(-3, 0)), "batch_size": int(rng.choice([16, 32, 64, 0])),
            "init_scale": float(10 ** rng.uniform(-2, 0.5)),
            "wd": float(rng.choice([0, 1e-4, 1e-2])), "epochs": 30}


def generate(n_configs=200, seeds=(0, 1, 2), out_dir=".", rng_seed=123):
    """Returns (runs, steps, forward) DataFrames and writes them to out_dir as parquet."""
    rng = np.random.RandomState(rng_seed)
    runs, steps, fwd = [], [], []
    for c in range(n_configs):
        cfg = sample_config(rng)
        for seed in seeds:
            run_id = f"c{c}_s{seed}"
            s, f, final = train(cfg, seed, run_id)
            runs.append({"run_id": run_id, "config_id": c, "seed": seed, **cfg, **final})
            steps += s
            fwd += f
    runs, steps, fwd = pd.DataFrame(runs), pd.DataFrame(steps), pd.DataFrame(fwd)
    for name, df in [("runs", runs), ("steps", steps), ("forward", fwd)]:
        df.to_parquet(f"{out_dir}/{name}.parquet")
    return runs, steps, fwd
