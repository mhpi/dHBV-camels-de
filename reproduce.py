"""Reproduce dHBV vs LSTM metrics + boxplots on CAMELS-DE.

Single-file release: loads pretrained networks (two differentiable-HBV
checkpoints and one LSTM with catchment embedding), runs PyTorch forward on
the 1,347-basin CAMELS-DE test split, computes per-catchment NSE / percent
bias / low-flow RMSE / high-flow RMSE, and writes a 4-panel boxplot PNG plus
summary CSVs.

Expected median NSE on the test window 2011-01-01 .. 2020-12-31:
    dHBV-base 0.838,  dHBV-dyn_uzl 0.869,  torch.nn.LSTM 0.866.

Usage
-----
    pip install -r requirements.txt
    python reproduce.py --root . --out-dir results/

CLI flags
---------
    --root                 release directory (contains weights/, data/, config/)
    --out-dir              output directory (default: results/)
    --device               cuda or cpu (default: cuda)
    --models               comma-separated subset of {dhbv_base, dhbv_dyn_uzl, lstm}
                           default: all three
    --dhbv-epoch           which dHBV epoch to evaluate (50 or 100; default 100,
                           dyn_uzl is only available at ep100)

Layout
------
    release/
    ├── weights/
    │   ├── dhbv_base/    hbv_1_1p_ep50.pt + hbv_1_1p_ep100.pt
    │   ├── dhbv_dyn_uzl/ hbv_1_1p_ep100.pt
    │   └── lstm/         embedding.pt + decoder.pt + hparams.json
    ├── config/
    │   ├── config_dhbv_base.yaml
    │   └── config_dhbv_dyn_uzl.yaml
    └── data/
        ├── camels_de.pkl + gage_info.npy   (dHBV inputs; see data/README.md)
        └── data_test_CAMELS_DE1.00.csv     (LSTM inputs)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn

SEASON_MONTHS = {
    "DJF": [12, 1, 2], "MAM": [3, 4, 5], "JJA": [6, 7, 8], "SON": [9, 10, 11],
    "ALL": list(range(1, 13)),
}
SEASONS = list(SEASON_MONTHS)
COLORS = {"dHBV-base": "#d95f02", "dHBV-dyn_uzl": "#1b9e77", "torch.nn.LSTM": "#7570b3"}


# ────────────────────────── dHBV forward (via dmg) ──────────────────────────
def _load_config(path: str) -> dict:
    """Inlined version of `example.load_config` so users don't need a separate
    `generic_deltamodel` clone on PYTHONPATH. Uses `initialize_config_dir`
    with the absolute path so resolution is independent of CWD."""
    import os
    import hydra
    from dmg.core.utils import initialize_config

    abs_dir = os.path.abspath(os.path.dirname(path))
    config_name = os.path.splitext(os.path.basename(path))[0]
    with hydra.initialize_config_dir(config_dir=abs_dir, version_base="1.3"):
        cfg = hydra.compose(config_name=config_name)
    return initialize_config(cfg, make_dirs=False)


def run_dhbv(weights_dir: str, config_path: str, data_pkl: str, gage_info: str,
             scratch_dir: str, test_epoch: int = 100, device: str = "cuda"
             ) -> Tuple[np.ndarray, np.ndarray]:
    """Forward of one dHBV variant; return (pred, obs) shaped (N, T)."""
    from dmg import ModelHandler
    from dmg.core.utils import import_data_loader, import_trainer, set_randomseed

    cfg = _load_config(config_path)
    cfg["mode"] = "test"
    cfg["device"] = device
    cfg["observations"]["data_path"] = data_pkl
    cfg["observations"]["gage_info"] = gage_info
    cfg["output_dir"] = scratch_dir
    cfg["model_dir"] = weights_dir
    cfg["sim_dir"] = os.path.join(scratch_dir, "sim")
    cfg["test"]["test_epoch"] = test_epoch
    os.makedirs(cfg["sim_dir"], exist_ok=True)

    set_randomseed(cfg["seed"])
    dl = import_data_loader(cfg["data_loader"])(cfg, test_split=True, overwrite=False)
    model = ModelHandler(cfg, verbose=False)
    trainer = import_trainer(cfg["trainer"])(cfg, model, eval_dataset=dl.eval_dataset, verbose=False)
    trainer.evaluate()
    pred = trainer.predictions["streamflow"].squeeze().T  # (T,N,1) -> (N,T)
    obs = dl.eval_dataset["target"].cpu().numpy().squeeze().T
    if obs.shape[1] > pred.shape[1]:
        obs = obs[:, -pred.shape[1]:]
    return pred.astype(np.float32), obs.astype(np.float32)


# ────────────────────────── LSTM forward (inline) ───────────────────────────
class _TimeDistributed(nn.Module):
    """Wraps a module so per-timestep application keeps a `.m` attribute,
    matching the state_dict key layout of the released checkpoint."""

    def __init__(self, module: nn.Module):
        super().__init__()
        self.m = module

    def forward(self, x):
        return self.m(x)


class _LSTMDecoder(nn.Module):
    """Catchment-embedding + LSTM decoder used as the LSTM baseline.

    The FC head is wrapped in `_TimeDistributed` to match the released
    checkpoint key layout (`fc.m.0/3/6.weight` etc.).
    """

    def __init__(self, latent_dim: int, feature_dim: int, lstm_hidden_dim: int,
                 fc_sizes: list[int], num_lstm_layers: int = 1, p: float = 0.0):
        super().__init__()
        self.lstm = nn.LSTM(
            latent_dim + feature_dim, lstm_hidden_dim, num_layers=num_lstm_layers,
            dropout=p if num_lstm_layers > 1 else 0.0, batch_first=True,
        )
        layers = []
        prev = lstm_hidden_dim
        for s in fc_sizes:
            layers += [nn.Linear(prev, s), nn.ReLU(), nn.Dropout(p)]
            prev = s
        layers += [nn.Linear(prev, 1)]
        self.fc = _TimeDistributed(nn.Sequential(*layers))

    def decode(self, code: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        code_t = code.unsqueeze(1).expand(-1, T, -1)
        h, _ = self.lstm(torch.cat([code_t, x], dim=2))
        return self.fc(h).squeeze(-1)[:, 365:]


def run_lstm(weights_dir: str, test_csv: str, device: str = "cuda"
             ) -> Tuple[np.ndarray, np.ndarray]:
    hp = json.load(open(os.path.join(weights_dir, "hparams.json")))
    df = pd.read_csv(test_csv)
    n_catch = df["catchment_name"].nunique()
    record_length = len(df) // n_catch
    assert n_catch * record_length == len(df), "irregular test CSV"
    x_all = df[["P", "T", "PET"]].to_numpy(dtype=np.float32).reshape(n_catch, record_length, 3)
    y_all = df["Q"].to_numpy(dtype=np.float32).reshape(n_catch, record_length)

    emb = nn.Embedding(n_catch, hp["latent_dim"]).to(device)
    dec = _LSTMDecoder(hp["latent_dim"], hp["feature_dim"], hp["lstm_hidden_dim"],
                       hp["fc_sizes"], num_lstm_layers=hp["n_lstm_layers"],
                       p=hp["dropout_rate"]).to(device)
    emb.load_state_dict(torch.load(os.path.join(weights_dir, "embedding.pt"), map_location=device))
    dec.load_state_dict(torch.load(os.path.join(weights_dir, "decoder.pt"), map_location=device))
    emb.eval(); dec.eval()

    preds = []
    with torch.no_grad():
        for s in range(0, n_catch, 64):
            e = min(s + 64, n_catch)
            xb = torch.from_numpy(x_all[s:e]).to(device)
            ids = torch.arange(s, e, device=device)
            preds.append(dec.decode(emb(ids), xb).cpu().numpy())
    pred = np.concatenate(preds, axis=0).astype(np.float32)
    obs = y_all[:, hp["base_length"]:].astype(np.float32)
    return pred, obs


# ───────────────────────────── metrics ─────────────────────────────
def per_catchment_nse(pred, obs):
    out = np.full(pred.shape[0], np.nan, dtype=np.float32)
    for i in range(pred.shape[0]):
        m = ~np.isnan(obs[i])
        if m.sum() < 30: continue
        o = obs[i, m]; p = pred[i, m]
        ss = ((o - o.mean()) ** 2).sum()
        if ss == 0: continue
        out[i] = 1.0 - ((o - p) ** 2).sum() / ss
    return out


def per_catchment_pbias(pred, obs):
    out = np.full(pred.shape[0], np.nan, dtype=np.float32)
    for i in range(pred.shape[0]):
        m = ~np.isnan(obs[i])
        if m.sum() < 30: continue
        o = obs[i, m]; p = pred[i, m]
        om = o.mean()
        if om <= 0: continue
        out[i] = 100.0 * (p.mean() - om) / om
    return out


def per_catchment_flow_rmse(pred, obs, q_lo, q_hi):
    out = np.full(pred.shape[0], np.nan, dtype=np.float32)
    for i in range(pred.shape[0]):
        m = ~np.isnan(obs[i])
        if m.sum() < 30: continue
        o = obs[i, m]; p = pred[i, m]
        lo = np.quantile(o, q_lo); hi = np.quantile(o, q_hi)
        sel = (o >= lo) & (o <= hi)
        if sel.sum() < 5: continue
        out[i] = np.sqrt(((p[sel] - o[sel]) ** 2).mean())
    return out


# ───────────────────────────── plotting ────────────────────────────
def make_plots(model_data: dict, out_dir: str, model_order: list):
    n_pred = next(iter(model_data.values()))[0].shape[1]
    dates = pd.date_range("2011-01-01", periods=n_pred, freq="D")
    months = dates.month
    masks = {s: np.isin(months, SEASON_MONTHS[s]) for s in SEASONS}

    nse_rows = {s: {} for s in SEASONS}
    pb_rows = {s: {} for s in SEASONS}
    for s in SEASONS:
        for name, (pred, obs) in model_data.items():
            nse_rows[s][name] = per_catchment_nse(pred[:, masks[s]], obs[:, masks[s]])
            pb_rows[s][name] = per_catchment_pbias(pred[:, masks[s]], obs[:, masks[s]])
    rmse_lo = {m: per_catchment_flow_rmse(*model_data[m], 0.0, 0.30) for m in model_order}
    rmse_hi = {m: per_catchment_flow_rmse(*model_data[m], 0.98, 1.0) for m in model_order}

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    def grouped(ax, rows, ylabel, title, ylim=None, hline=None):
        n_g = len(SEASONS); width = 0.22
        offsets = np.linspace(-(len(model_order) - 1) / 2, (len(model_order) - 1) / 2,
                              len(model_order)) * (width * 1.05)
        for j, m in enumerate(model_order):
            positions = np.arange(n_g) + offsets[j]
            arrs = [rows[s][m][~np.isnan(rows[s][m])] for s in SEASONS]
            bp = ax.boxplot(arrs, positions=positions, widths=width, patch_artist=True,
                            showfliers=False, medianprops=dict(color="black", lw=1.4))
            for p in bp["boxes"]:
                p.set_facecolor(COLORS[m]); p.set_alpha(0.85)
        ax.set_xticks(np.arange(n_g)); ax.set_xticklabels(SEASONS)
        ax.set_ylabel(ylabel); ax.set_title(title)
        if ylim is not None: ax.set_ylim(*ylim)
        if hline is not None: ax.axhline(hline, color="grey", lw=0.8, ls="--")
        ax.grid(True, axis="y", alpha=0.3)

    def single(ax, d, ylabel, title, ylog=False):
        arrs = [d[m][~np.isnan(d[m])] for m in model_order]
        bp = ax.boxplot(arrs, positions=np.arange(len(model_order)), widths=0.55,
                        patch_artist=True, showfliers=False,
                        medianprops=dict(color="black", lw=1.4))
        for p, m in zip(bp["boxes"], model_order):
            p.set_facecolor(COLORS[m]); p.set_alpha(0.85)
        ax.set_xticks(np.arange(len(model_order))); ax.set_xticklabels(model_order, rotation=15)
        ax.set_ylabel(ylabel); ax.set_title(title)
        if ylog: ax.set_yscale("log")
        ax.grid(True, axis="y", alpha=0.3)

    grouped(axes[0, 0], nse_rows, "NSE", "Per-catchment NSE by season", ylim=(0.0, 1.0))
    grouped(axes[0, 1], pb_rows, "Percent bias (%)", "Per-catchment % bias by season",
            ylim=(-50, 50), hline=0.0)
    single(axes[1, 0], rmse_lo, "RMSE (mm/d) on bottom 30% obs",
           "Per-catchment low-flow RMSE", ylog=True)
    single(axes[1, 1], rmse_hi, "RMSE (mm/d) on top 2% obs",
           "Per-catchment high-flow RMSE")

    handles = [plt.Rectangle((0, 0), 1, 1, fc=COLORS[m], alpha=0.85, ec="black") for m in model_order]
    fig.legend(handles, model_order, loc="upper center", ncol=len(model_order),
               bbox_to_anchor=(0.5, 1.02), framealpha=0.95)
    fig.suptitle("CAMELS-DE: dHBV vs LSTM (1,347 basins, 2011-2020)", y=1.04, fontsize=13)
    plt.tight_layout()
    fig_path = os.path.join(out_dir, "metrics_boxplots.png")
    plt.savefig(fig_path, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {fig_path}")

    rows = []
    for s in SEASONS:
        for m in model_order:
            nse = nse_rows[s][m]; pb = pb_rows[s][m]
            nse_c = nse[~np.isnan(nse)]; pb_c = pb[~np.isnan(pb)]
            rows.append({"season": s, "model": m,
                         "nse_median": float(np.median(nse_c)),
                         "nse_p25": float(np.quantile(nse_c, 0.25)),
                         "nse_p75": float(np.quantile(nse_c, 0.75)),
                         "pbias_median_pct": float(np.median(pb_c)),
                         "abs_pbias_median_pct": float(np.median(np.abs(pb_c)))})
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "seasonal_summary.csv"), index=False)
    flow_rows = []
    for m in model_order:
        lo = rmse_lo[m][~np.isnan(rmse_lo[m])]
        hi = rmse_hi[m][~np.isnan(rmse_hi[m])]
        flow_rows.append({"model": m,
                          "rmse_low_median": float(np.median(lo)),
                          "rmse_high_median": float(np.median(hi))})
    pd.DataFrame(flow_rows).to_csv(os.path.join(out_dir, "flow_rmse_summary.csv"), index=False)
    print("wrote seasonal_summary.csv + flow_rmse_summary.csv")


# ───────────────────────────── main ────────────────────────────────
DHBV_VARIANTS = {
    "dhbv_base":    {"name": "dHBV-base",    "config": "config_dhbv_base.yaml",
                     "weights_subdir": "dhbv_base",    "epoch": 50},
    "dhbv_dyn_uzl": {"name": "dHBV-dyn_uzl", "config": "config_dhbv_dyn_uzl.yaml",
                     "weights_subdir": "dhbv_dyn_uzl", "epoch": 100},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="release dir")
    ap.add_argument("--out-dir", default="./results")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--models", default="dhbv_base,dhbv_dyn_uzl,lstm",
                    help="comma-separated subset of {dhbv_base,dhbv_dyn_uzl,lstm}")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    root = os.path.abspath(args.root)
    data_pkl = os.path.join(root, "data", "camels_de.pkl")
    gage_info = os.path.join(root, "data", "gage_info.npy")

    chosen = [m.strip() for m in args.models.split(",") if m.strip()]
    model_data = {}; model_order = []

    for key in ("dhbv_base", "dhbv_dyn_uzl"):
        if key not in chosen: continue
        v = DHBV_VARIANTS[key]
        print(f"\n=== {v['name']} forward (ep{v['epoch']}) ===")
        weights = os.path.join(root, "weights", v["weights_subdir"])
        cfg_path = os.path.join(root, "config", v["config"])
        pred, obs = run_dhbv(weights, cfg_path, data_pkl, gage_info,
                             os.path.join(args.out_dir, f"scratch_{key}"),
                             test_epoch=v["epoch"], device=args.device)
        np.save(os.path.join(args.out_dir, f"{key}_streamflow.npy"), pred)
        model_data[v["name"]] = (pred, obs)
        model_order.append(v["name"])

    if "lstm" in chosen:
        print("\n=== torch.nn.LSTM forward ===")
        pred, obs = run_lstm(os.path.join(root, "weights", "lstm"),
                             os.path.join(root, "data", "data_test_CAMELS_DE1.00.csv"),
                             device=args.device)
        np.save(os.path.join(args.out_dir, "lstm_streamflow.npy"), pred)
        model_data["torch.nn.LSTM"] = (pred, obs)
        model_order.append("torch.nn.LSTM")

    if model_order:
        first_obs = next(iter(model_data.values()))[1]
        np.save(os.path.join(args.out_dir, "streamflow_obs.npy"), first_obs)
        print("\n=== plots + metrics ===")
        make_plots(model_data, args.out_dir, model_order)
    print(f"\nDone. Outputs in {args.out_dir}")


if __name__ == "__main__":
    main()
