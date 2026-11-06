"""Small, explicit model search spaces; all tuning uses validation only."""
from copy import deepcopy
import numpy as np
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor

MODEL_NAMES = ["Persistence", "Ridge", "MLP", "MLP_noise", "RF", "XGBoost"]


class Persistence:
    def predict(self, x):
        return np.asarray(x)[:, :9].copy()


class TorchMLP:
    """Eq. (6) in Vasisht et al.; 20 -> 32 -> 32 -> 9, direct T(t+1)."""
    def __init__(self, seed=0, learning_rate=0.003, noise_sigma=0., epochs=1500):
        self.seed, self.learning_rate = seed, learning_rate
        self.noise_sigma, self.epochs = noise_sigma, epochs

    def fit(self, x, y, xv, yv):
        torch.manual_seed(self.seed)
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        self.x_scaler, self.y_scaler = StandardScaler(), StandardScaler()
        xt = torch.tensor(self.x_scaler.fit_transform(x), dtype=torch.float32)
        yt = torch.tensor(self.y_scaler.fit_transform(y), dtype=torch.float32)
        vt = torch.tensor(self.x_scaler.transform(xv), dtype=torch.float32)
        self.net = nn.Sequential(nn.Linear(20, 32), nn.Tanh(), nn.Linear(32, 32),
                                 nn.Tanh(), nn.Linear(32, 9))
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate,
                                     weight_decay=1e-4)
        sigma = torch.tensor(self.noise_sigma / self.x_scaler.scale_[:9],
                             dtype=torch.float32)
        best, stale, state = float("inf"), 0, None
        self.history = []
        for epoch in range(1, self.epochs+1):
            self.net.train()
            noisy = xt.clone()
            if self.noise_sigma:
                noisy[:, :9] += torch.randn_like(noisy[:, :9]) * sigma
            optimizer.zero_grad()
            loss = torch.mean((self.net(noisy)-yt)**2)
            loss.backward()
            optimizer.step()
            if epoch % 10 == 0 or epoch == self.epochs:
                self.net.eval()
                with torch.no_grad():
                    pred = self.y_scaler.inverse_transform(self.net(vt).numpy())
                score = float(np.sqrt(np.mean((pred-yv)**2)))
                self.history.append({"epoch": epoch, "train_scaled_mse": float(loss.detach()),
                                     "validation_rmse_C": score})
                if score < best-1e-7:
                    best, stale = score, 0
                    self.best_epoch = epoch
                    state = deepcopy(self.net.state_dict())
                else:
                    stale += 10
                if stale >= 200:
                    break
        if state is None:
            raise RuntimeError("MLP did not produce a finite validation checkpoint")
        self.net.load_state_dict(state)
        self.net.eval()
        return self

    def predict(self, x):
        with torch.no_grad():
            xt = torch.tensor(self.x_scaler.transform(x), dtype=torch.float32)
            return self.y_scaler.inverse_transform(self.net(xt).numpy())


def candidates(name, seed, epochs=1500):
    if name == "Persistence":
        return [({}, Persistence())]
    if name == "Ridge":
        return [({"alpha": a}, make_pipeline(StandardScaler(), Ridge(alpha=a)))
                for a in (0.1, 1., 10., 100.)]
    if name.startswith("MLP"):
        sigma = 0.2 if name == "MLP_noise" else 0.
        return [({"lr": lr, "sigma_C": sigma, "epochs_max": epochs},
                 TorchMLP(seed, lr, sigma, epochs)) for lr in (0.003, 0.01)]
    if name == "RF":
        return [({"n_estimators": 200, "max_depth": depth, "min_samples_leaf": leaf},
                 RandomForestRegressor(n_estimators=200, max_depth=depth,
                     min_samples_leaf=leaf, max_features=1.0, n_jobs=1,
                     random_state=seed)) for depth, leaf in ((6, 2), (None, 3))]
    if name == "XGBoost":
        return [({"n_estimators": 250, "max_depth": depth, "learning_rate": 0.03,
                  "reg_lambda": 5., "subsample": 0.9, "colsample_bytree": 1.0},
                 MultiOutputRegressor(XGBRegressor(n_estimators=250,
                    max_depth=depth, learning_rate=0.03, reg_lambda=5.,
                    subsample=0.9, colsample_bytree=1., objective="reg:squarederror",
                    tree_method="hist", n_jobs=1, random_state=seed), n_jobs=1))
                for depth in (2, 3)]
    raise ValueError(name)


def select_model(name, seed, train, valid, epochs=1500):
    x, y = train
    xv, yv = valid
    best_score, best_model, best_parameters = float("inf"), None, None
    trials = []
    for parameters, model in candidates(name, seed, epochs):
        if isinstance(model, TorchMLP):
            model.fit(x, y, xv, yv)
            parameters["best_epoch"] = model.best_epoch
        elif not isinstance(model, Persistence):
            model.fit(x, y)
        score = float(np.sqrt(np.mean((model.predict(xv)-yv)**2)))
        trials.append({"parameters": parameters, "validation_rmse_C": score,
                       "history": getattr(model, "history", [])})
        if score < best_score:
            best_score, best_model, best_parameters = score, model, parameters
    if best_model is None:
        raise RuntimeError(f"No finite validation result for {name}")
    return best_model, best_score, best_parameters, trials
