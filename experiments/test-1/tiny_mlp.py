"""Tiny 2-H-1 MLP in plain numpy, so every weight, gradient and activation can be logged."""
import numpy as np
from sklearn.datasets import make_moons, make_circles

ACTS = {  # name: (function, derivative wrt pre-activation given activation output a)
    "tanh": (np.tanh, lambda z, a: 1 - a ** 2),
    "relu": (lambda z: np.maximum(z, 0), lambda z, a: (z > 0).astype(float)),
}


def make_data(name, noise, n, seed):
    """2D binary classification data. Returns X (n,2), y (n,)."""
    if name == "moons":
        return make_moons(n, noise=noise, random_state=seed)
    if name == "circles":
        return make_circles(n, noise=noise, factor=0.5, random_state=seed)
    rng = np.random.RandomState(seed)  # xor
    X = rng.uniform(-1, 1, (n, 2))
    y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int)
    return X + rng.normal(0, noise, X.shape), y


def param_names(H):
    """Flat parameter order: W1 (H x 2), b1 (H), W2 (H), b2."""
    return ([f"W1_{i}{j}" for i in range(H) for j in range(2)]
            + [f"b1_{i}" for i in range(H)] + [f"W2_{i}" for i in range(H)] + ["b2"])


def unpack(p, H):
    return p[:2 * H].reshape(H, 2), p[2 * H:3 * H], p[3 * H:4 * H], p[4 * H]


def forward(p, X, H, act):
    """Returns pre-activations z1, activations a1, output logit z2, probability."""
    W1, b1, W2, b2 = unpack(p, H)
    z1 = X @ W1.T + b1
    a1 = ACTS[act][0](z1)
    z2 = a1 @ W2 + b2
    return z1, a1, z2, 1 / (1 + np.exp(-z2))


def loss_acc(p, X, y, H, act):
    prob = forward(p, X, H, act)[3]
    eps = 1e-9
    loss = -np.mean(y * np.log(prob + eps) + (1 - y) * np.log(1 - prob + eps))
    return loss, np.mean((prob > 0.5) == y)


def grad(p, X, y, H, act, wd):
    """Manual backprop for binary cross-entropy (+ L2 weight decay)."""
    W1, b1, W2, b2 = unpack(p, H)
    z1, a1, z2, prob = forward(p, X, H, act)
    dz2 = (prob - y) / len(y)                       # dL/dz2
    dW2, db2 = a1.T @ dz2, dz2.sum()
    dz1 = np.outer(dz2, W2) * ACTS[act][1](z1, a1)  # dL/dz1
    dW1, db1 = dz1.T @ X, dz1.sum(0)
    g = np.concatenate([dW1.ravel(), db1, dW2, [db2]])
    return g + wd * p


class Optimizer:
    """sgd / momentum / adam on a flat parameter vector."""
    def __init__(self, kind, lr, size):
        self.kind, self.lr, self.t = kind, lr, 0
        self.m, self.v = np.zeros(size), np.zeros(size)

    def step(self, p, g):
        self.t += 1
        if self.kind == "sgd":
            return p - self.lr * g
        if self.kind == "momentum":
            self.m = 0.9 * self.m + g
            return p - self.lr * self.m
        self.m = 0.9 * self.m + 0.1 * g
        self.v = 0.999 * self.v + 0.001 * g ** 2
        mh, vh = self.m / (1 - 0.9 ** self.t), self.v / (1 - 0.999 ** self.t)
        return p - self.lr * mh / (np.sqrt(vh) + 1e-8)


def train(cfg, seed, run_id, n_probe=8, probe_every=10):
    """Train one run. Returns (steps_rows, forward_rows, final_metrics).

    steps_rows   : one dict per optimizer step - weights before, gradient, weights after, metrics.
    forward_rows : the probe points pushed through the network (inputs, mixes z, activations a, output).
    """
    H, act = cfg["H"], cfg["act"]
    Xtr, ytr = make_data(cfg["data"], cfg["noise"], 200, seed=0)   # same data for every seed
    Xva, yva = make_data(cfg["data"], cfg["noise"], 200, seed=1)
    probe_X, probe_y = Xva[:n_probe], yva[:n_probe]
    rng = np.random.RandomState(seed)                              # seed only controls init + batch order
    p = rng.normal(0, cfg["init_scale"], 4 * H + 1)
    opt = Optimizer(cfg["opt"], cfg["lr"], p.size)
    names = param_names(H)
    bs = len(Xtr) if cfg["batch_size"] == 0 else cfg["batch_size"]  # 0 = full batch

    steps, fwd, step = [], [], 0
    for epoch in range(cfg["epochs"]):
        order = rng.permutation(len(Xtr))
        for i in range(0, len(Xtr), bs):
            idx = order[i:i + bs]
            g = grad(p, Xtr[idx], ytr[idx], H, act, cfg["wd"])
            p_new = opt.step(p, g)
            vl, va = loss_acc(p_new, Xva, yva, H, act)
            row = {"run_id": run_id, "epoch": epoch, "step": step,
                   "batch_loss": loss_acc(p, Xtr[idx], ytr[idx], H, act)[0],  # loss before this update
                   "val_loss": vl, "val_acc": va}
            for n, b, gr, a in zip(names, p, g, p_new):
                row[f"before_{n}"], row[f"grad_{n}"], row[f"after_{n}"] = b, gr, a
            steps.append(row)

            if step % probe_every == 0:  # full forward pass on probe points (weights before the update)
                z1, a1, z2, prob = forward(p, probe_X, H, act)
                for k in range(n_probe):
                    r = {"run_id": run_id, "step": step, "x1": probe_X[k, 0], "x2": probe_X[k, 1],
                         "y": probe_y[k], "z_out": z2[k], "y_hat": prob[k]}
                    for h in range(H):
                        r[f"z_h{h}"], r[f"a_h{h}"] = z1[k, h], a1[k, h]
                    fwd.append(r)
            p, step = p_new, step + 1

    vl, va = loss_acc(p, Xva, yva, H, act)
    tl, _ = loss_acc(p, Xtr, ytr, H, act)
    final = {"final_train_loss": tl, "final_val_loss": vl, "final_val_acc": va, "n_steps": step}
    return steps, fwd, final
