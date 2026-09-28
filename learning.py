"""Measured corrosion data -> cross-validated GP -> EI recommendations."""
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.spatial import cKDTree
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, RBF
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from core import KEYS, key, strict_csv, save_csv

TARGETS = {
    "icorr": ("icorr_mean_A_cm2", "icorr_sd_A_cm2", "log", .03),
    "Epit": ("Epit_mean_V_AgAgCl", "Epit_sd_V", "negative", .005),
    "Rct": ("Rct_mean_ohm_cm2", "Rct_sd_ohm_cm2", "negative_log", .03),
}


def measured(path, cfg):
    df = strict_csv(path)
    required = KEYS + ["n_replicates", "experiment_context"] + [c for v in list(TARGETS.values())[:2] for c in v[:2]]
    missing = set(required)-set(df)
    if missing:
        raise ValueError(f"Missing experiment columns: {sorted(missing)}")
    if not df.experiment_context.eq(cfg["experiment_context"]).all():
        raise ValueError("Different electrolyte/protocol contexts must not be pooled")
    for col in KEYS + ["n_replicates"]:
        df[col] = pd.to_numeric(df[col], errors="raise")
    if not np.isfinite(df[KEYS+["n_replicates"]].to_numpy()).all() or (df[KEYS] < 0).any().any():
        raise ValueError("Invalid experiment compositions/counts")
    if (df.n_replicates < 1).any() or (df.n_replicates % 1 != 0).any():
        raise ValueError("n_replicates must be positive integers")
    pairs = [key(*r) for r in df[KEYS].to_numpy()]
    if len(set(pairs)) != len(pairs):
        raise ValueError("Use one aggregated mean/SD row per composition; do not double-count replicates")
    if len(df) < 6:
        raise ValueError("At least 6 distinct measured compositions required (8-12 recommended)")
    for col, spec in zip(KEYS, [cfg["ce_grid"], cfg["cr_grid"]]):
        if not df[col].between(spec[0]-1e-9, spec[1]+1e-9).all():
            raise ValueError(f"Measured {col} outside configured design space")
    return df


def target_data(df, name):
    mean_col, sd_col, transform, floor = TARGETS[name]
    values = df[[mean_col, sd_col]].apply(pd.to_numeric, errors="raise").to_numpy(float)
    if not np.isfinite(values).all() or (values[:, 1] < 0).any():
        raise ValueError(f"Invalid/missing {name} values; do not substitute zero")
    y, sd = values.T
    sem = sd/np.sqrt(df.n_replicates.to_numpy(float))
    if "log" in transform:
        if (y <= 0).any():
            raise ValueError(f"{name} must be positive for log10")
        noise = np.maximum(sem/(y*np.log(10)), floor)
        y = np.log10(y)
    else:
        noise = np.maximum(sem, floor)
    if transform.startswith("negative"):
        y = -y
    return y, noise


def fit_gp(x, y, noise, family, cfg):
    scaler = StandardScaler().fit(x)
    offset, scale = float(y.mean()), float(y.std())
    scale = max(scale, 1e-8)
    base = RBF(np.ones(2), (1e-2, 1e3)) if family == "RBF" else Matern(np.ones(2), (1e-2, 1e3), nu=2.5 if family == "Matern52" else 1.5)
    gp = GaussianProcessRegressor(kernel=ConstantKernel(1., (1e-3, 1e3))*base,
        alpha=(noise/scale)**2+1e-10, normalize_y=False,
        n_restarts_optimizer=cfg["optimizer_restarts"], random_state=cfg["seed"])
    gp.fit(scaler.transform(x), (y-offset)/scale)
    return scaler, gp, offset, scale


def predict(model, x):
    scaler, gp, offset, scale = model
    mu, sd = gp.predict(scaler.transform(x), return_std=True)
    return mu*scale+offset, sd*scale  # latent uncertainty: observation noise excluded


def choose_model(x, y, noise, cfg):
    records = []
    for family in ["Matern52", "RBF", "Matern32"]:
        errors, variances = [], []
        for train, test in LeaveOneOut().split(x):
            model = fit_gp(x[train], y[train], noise[train], family, cfg)
            mu, sd = predict(model, x[test])
            errors.append(float(y[test][0]-mu[0]))
            variances.append(float(sd[0]**2+noise[test][0]**2))
        errors, variances = np.array(errors), np.maximum(variances, 1e-16)
        records.append(dict(kernel=family, rmse=float(np.sqrt(np.mean(errors**2))),
            nlpd=float(np.mean(.5*np.log(2*np.pi*variances)+.5*errors**2/variances)),
            coverage95=float(np.mean(np.abs(errors) <= 1.96*np.sqrt(variances)))))
    # Proper predictive score jointly considers error and calibrated uncertainty.
    scores = pd.DataFrame(records).sort_values(["nlpd", "rmse"])
    winner = scores.iloc[0].kernel
    return fit_gp(x, y, noise, winner, cfg), scores


def expected_improvement(mu, sd, best):
    mu, sd = np.asarray(mu), np.asarray(sd)
    improvement = best-mu
    result = np.maximum(improvement, 0.)
    mask = sd > 1e-14
    z = improvement[mask]/sd[mask]
    result[mask] = improvement[mask]*norm.cdf(z)+sd[mask]*norm.pdf(z)
    return result


def unit_range(values):
    values = np.asarray(values, float)
    return (values-values.min())/np.ptp(values) if np.ptp(values) > 1e-14 else np.zeros_like(values)


def recommend(pool, experiment_file, cfg, out):
    exp = measured(experiment_file, cfg)
    seen = {key(*r) for r in exp[KEYS].to_numpy()}
    unmeasured = pool.loc[[key(*r) not in seen for r in pool[KEYS].to_numpy()]].drop_duplicates(KEYS).copy().reset_index(drop=True)
    if len(unmeasured) < 3:
        raise ValueError("Fewer than three unmeasured feasible candidates remain")
    x, xp = exp[KEYS].to_numpy(float), unmeasured[KEYS].to_numpy(float)
    score_tables, uncertainty = [], []
    targets = ["icorr", "Epit"]
    rct_cols = list(TARGETS["Rct"][:2])
    if set(rct_cols).issubset(exp) and exp[rct_cols].notna().any().any():
        targets.append("Rct")
    for name in targets:
        y, noise = target_data(exp, name)
        model, scores = choose_model(x, y, noise, cfg)
        scores["target"] = name
        score_tables.append(scores)
        mu, sd = predict(model, xp)
        unmeasured[f"objective_mu_{name}"] = mu
        unmeasured[f"objective_std_{name}"] = sd
        # Noise-aware plug-in incumbent: best posterior mean at observed designs.
        incumbent = float(predict(model, x)[0].min())
        unmeasured[f"EI_{name}"] = expected_improvement(mu, sd, incumbent)
        if name == "icorr":
            unmeasured["pred_icorr_median_A_cm2"] = 10**mu
            unmeasured["pred_icorr_mean_A_cm2"] = np.exp(np.log(10)*mu+.5*(np.log(10)*sd)**2)
        elif name == "Epit":
            unmeasured["pred_Epit_V_AgAgCl"] = -mu
        else:
            unmeasured["pred_Rct_median_ohm_cm2"] = 10**(-mu)
        uncertainty.append(unit_range(sd))
    # cKDTree avoids the original O(N^2) full-pool distance loop.
    scaled = xp/np.maximum(np.ptp(xp, axis=0), 1e-12)
    d, neighbors = cKDTree(scaled).query(scaled, k=min(5, len(scaled)))
    q = unmeasured.matrix_Cr_wt.to_numpy(float)
    boundary = np.max(np.abs(q[:, None]-q[neighbors[:, 1:]])/np.maximum(d[:, 1:], 1e-12), axis=1)
    unmeasured["boundary_proxy"] = boundary
    unmeasured["exploration_score"] = .7*np.mean(uncertainty, axis=0)+.3*unit_range(boundary)
    used, chosen = set(), []
    for role, column in [("A_EI_icorr", "EI_icorr"), ("B_EI_Epit", "EI_Epit"), ("C_uncertainty_boundary", "exploration_score")]:
        for idx in unmeasured.sort_values(column, ascending=False, kind="stable").index:
            if idx not in used:
                row = unmeasured.loc[idx].copy()
                row["selection_type"] = role
                chosen.append(row)
                used.add(idx)
                break
    selected = pd.DataFrame(chosen)
    save_csv(unmeasured, out/"predictions.csv")
    save_csv(selected, out/"next_candidates.csv")
    save_csv(pd.concat(score_tables), out/"kernel_comparison.csv")
    plot(unmeasured, exp, selected, out)
    return selected


def plot(pool, exp, selected, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, column in zip(axes, ["objective_mu_icorr", "pred_Epit_V_AgAgCl"]):
        points = ax.scatter(pool.Cr_wt, pool.Ce_wt, c=pool[column], s=7, cmap="viridis")
        ax.scatter(exp.Cr_wt, exp.Ce_wt, c="white", edgecolor="black", s=35)
        ax.scatter(selected.Cr_wt, selected.Ce_wt, c="red", marker="*", s=120)
        for _, row in selected.iterrows():
            ax.annotate(row.selection_type[0], (row.Cr_wt, row.Ce_wt))
        ax.set(xlabel="Cr (wt%)", ylabel="Ce (wt%)", title=column)
        fig.colorbar(points, ax=ax)
    fig.tight_layout()
    fig.savefig(out/"prediction_maps.png", dpi=180)
    plt.close(fig)
