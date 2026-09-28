"""Units: composition wt%, temperature deg C unless explicitly suffixed K."""
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

KEYS = ["Ce_wt", "Cr_wt"]


def load_settings(path):
    cfg = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if set(cfg["base_wt"]) & {"FE", "CE", "CR"}:
        raise ValueError("base_wt must exclude balance FE and variable CE/CR")
    for grid in (cfg["ce_grid"], cfg["cr_grid"]):
        start, end, step = grid
        if not np.isfinite(grid).all() or not 0 <= start <= end < 100 or step <= 0:
            raise ValueError("Invalid grid [inclusive start, inclusive end, step]")
    comp(cfg, cfg["ce_grid"][1], cfg["cr_grid"][1])
    if cfg["temperature_C"] <= -273.15 or cfg["pressure_Pa"] <= 0:
        raise ValueError("Invalid temperature or pressure")
    if cfg.get("process", {}).get("solution_treatment_C", cfg["temperature_C"]) != cfg["temperature_C"]:
        raise ValueError("Equilibrium temperature must match solution_treatment_C")
    for k in ("min_matrix_fraction", "max_harmful_fraction"):
        if not 0 <= cfg[k] <= 1:
            raise ValueError(k)
    return cfg


def comp(cfg, ce, cr):
    values = {**cfg["base_wt"], "CE": float(ce), "CR": float(cr)}
    if not np.isfinite(list(values.values())).all() or min(values.values()) < 0 or sum(values.values()) >= 100:
        raise ValueError("All wt% must be finite/nonnegative; Fe balance must be positive")
    return values


def grid(cfg):
    def axis(spec):
        lo, hi, step = spec
        return np.round(lo + step * np.arange(int(np.floor((hi-lo)/step + 1e-7))+1), 10)
    return pd.MultiIndex.from_product([axis(cfg["ce_grid"]), axis(cfg["cr_grid"])], names=KEYS).to_frame(index=False)


def key(ce, cr):
    return f"{ce:.8f}|{cr:.8f}"


def strict_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        if len(header) != len(set(header)):
            raise ValueError("Duplicate CSV headers")
        for i, row in enumerate(reader, 2):
            if row and len(row) != len(header):
                raise ValueError(f"{path}: row {i}: expected {len(header)} fields, got {len(row)}")
    return pd.read_csv(path, encoding="utf-8-sig")


def save_csv(df, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    df.to_csv(tmp, index=False, encoding="utf-8-sig")
    tmp.replace(path)


def feasible(df, cfg):
    return (df.calculation_status.eq("ok")
            & df.matrix_phase_fraction.ge(cfg["min_matrix_fraction"])
            & df.harmful_phase_fraction.le(cfg["max_harmful_fraction"])
            & df.matrix_Cr_wt.ge(cfg["min_matrix_cr_wt"]))


def diverse(df, n, cfg):
    """Deterministic maximin coverage in normalized Ce/Cr space."""
    df = df.drop_duplicates(KEYS).reset_index(drop=True)
    if n <= 0 or len(df) < n:
        raise ValueError(f"Need {n} distinct candidates; available={len(df)}")
    ranges = np.array([cfg["ce_grid"][1]-cfg["ce_grid"][0], cfg["cr_grid"][1]-cfg["cr_grid"][0]])
    x = df[KEYS].to_numpy(float) / np.maximum(ranges, 1e-12)
    chosen = [0]
    distances = np.full(len(x), np.inf)
    for _ in range(n-1):
        distances = np.minimum(distances, np.sum((x-x[chosen[-1]])**2, axis=1))
        distances[chosen] = -1
        chosen.append(int(np.argmax(distances)))
    return df.iloc[chosen].copy()


def scheil_metrics(temperature_K, solid_fraction):
    t, fs = np.asarray(temperature_K, float), np.asarray(solid_fraction, float)
    if t.shape != fs.shape or t.ndim != 1 or len(t) < 2 or not (np.isfinite(t).all() and np.isfinite(fs).all()):
        raise ValueError("Invalid Scheil curve")
    order = np.argsort(-t, kind="stable")
    t, fs = t[order], fs[order]
    if np.any(np.diff(fs) < -1e-5) or fs.min() < -1e-6 or fs.max() > 1.000001:
        raise ValueError("Nonphysical/nonmonotonic solid fraction")
    if fs[0] >= 0.90:
        raise ValueError("Curve starts after 90% solid; increase start temperature")
    def first(q):
        hits = t[fs >= q]
        return float(hits[0]-273.15) if len(hits) else np.nan
    t90, t99 = first(.90), first(.99)
    return dict(T_90solid_C=t90, T_99solid_C=t99, dT90_99_C=t90-t99,
                scheil_start_returned_C=float(t.max()-273.15),
                scheil_end_C=float(t.min()-273.15),
                calculated_temperature_span_C=float(np.ptp(t)),
                max_solid_fraction=float(fs.max()), n_scheil_steps=len(t))


class Checkpoint:
    """SQLite commits each point; configuration changes require a new run directory."""
    def __init__(self, directory, cfg):
        import sqlite3
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        manifest = self.directory / "settings_used.json"
        if manifest.exists() and json.loads(manifest.read_text(encoding="utf-8")) != cfg:
            raise ValueError("Settings changed: choose a different --out directory")
        manifest.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
        self.db = sqlite3.connect(self.directory / "checkpoint.sqlite")
        self.db.execute("CREATE TABLE IF NOT EXISTS results (stage TEXT, pair TEXT, payload TEXT, PRIMARY KEY(stage,pair))")

    def records(self, stage):
        return {k: json.loads(v) for k, v in self.db.execute("SELECT pair,payload FROM results WHERE stage=?", (stage,))}

    def put(self, stage, record):
        self.db.execute("INSERT OR REPLACE INTO results VALUES (?,?,?)", (stage, key(record["Ce_wt"], record["Cr_wt"]), json.dumps(record)))
        self.db.commit()

    def close(self):
        self.db.close()
