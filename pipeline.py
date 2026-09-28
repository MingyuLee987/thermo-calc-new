"""CLI: check -> equilibrium -> initial -> scheil -> template -> recommend."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from core import Checkpoint, KEYS, diverse, feasible, grid, key, load_settings, save_csv, strict_csv


def run_tc(stage, candidates, cfg, out, limit=None, retry=False):
    try:
        from tc_python import TCPython, UnrecoverableCalculationException
    except ImportError as exc:
        raise RuntimeError("TC-Python SDK unavailable. Run in the Python environment configured for your licensed Thermo-Calc installation.") from exc
    import tc_backend
    checkpoint = Checkpoint(out, cfg)
    try:
        previous = checkpoint.records(stage)
        status = "calculation_status" if stage == "equilibrium" else "scheil_status"
        pending = [r for r in candidates[KEYS].to_dict("records") if key(r["Ce_wt"], r["Cr_wt"]) not in previous or (retry and previous[key(r["Ce_wt"], r["Cr_wt"])].get(status) != "ok")]
        if limit is not None:
            pending = pending[:limit]
        if not pending:
            print("No pending points; exporting saved results.")
        else:
            with TCPython() as session:
                session.set_cache_folder(str((out/"tc_cache").resolve()))
                system = tc_backend.build_system(session, cfg)
                for i, row in enumerate(pending, 1):
                    try:
                        if stage == "equilibrium":
                            row.update(tc_backend.equilibrium(system, cfg, row["Ce_wt"], row["Cr_wt"]))
                        else:
                            metrics, curve_data = tc_backend.scheil(system, cfg, row["Ce_wt"], row["Cr_wt"])
                            row.update(metrics)
                            curves = out/"scheil_curves"
                            curves.mkdir(exist_ok=True)
                            for label, curve in curve_data.items():
                                save_csv(pd.DataFrame(curve), curves/f"Ce{row['Ce_wt']:.8f}_Cr{row['Cr_wt']:.8f}_{label}.csv")
                    except UnrecoverableCalculationException:
                        # A crashed server/session cannot be reused. Restart command to resume.
                        raise
                    except Exception as exc:
                        row.update({status: "failed", status.replace("status", "error"): repr(exc)})
                    checkpoint.put(stage, row)
                    if i == 1 or i % 100 == 0 or i == len(pending):
                        print(f"{stage}: {i}/{len(pending)} {row[status]}", flush=True)
    finally:
        records = checkpoint.records(stage)
        checkpoint.close()
        if records:
            df = pd.DataFrame(records.values()).sort_values(KEYS)
            if stage == "equilibrium":
                for col in ["matrix_phase_fraction", "matrix_Cr_wt", "harmful_phase_fraction"]:
                    if col not in df:
                        df[col] = np.nan
                df["calphad_feasible"] = feasible(df, cfg)
                save_csv(df, out/"calphad_equilibrium.csv")
                save_csv(df[df.calphad_feasible], out/"calphad_feasible_pool.csv")
            else:
                save_csv(df, out/"scheil_descriptors.csv")


def read_pool(out, cfg):
    pool = strict_csv(out/"calphad_feasible_pool.csv")
    required = KEYS+["calculation_status", "matrix_phase_fraction", "harmful_phase_fraction", "matrix_Cr_wt"]
    if not set(required).issubset(pool):
        raise ValueError("Incomplete equilibrium pool")
    pool = pool.loc[feasible(pool, cfg)].drop_duplicates(KEYS).copy()
    if pool.empty:
        raise ValueError("No CALPHAD-feasible candidates; inspect equilibrium failures/criteria")
    return pool


def checked_candidates(path, pool):
    df = strict_csv(path)
    if not set(KEYS).issubset(df) or df.empty:
        raise ValueError("Candidate CSV needs nonempty Ce_wt,Cr_wt columns")
    allowed = {key(*r) for r in pool[KEYS].to_numpy()}
    if any(key(*r) not in allowed for r in df[KEYS].to_numpy()):
        raise ValueError("Candidates outside calculated feasible pool")
    return df.drop_duplicates(KEYS)


def manufacturing_pool(pool, out, cfg):
    cap = cfg["max_dT90_99_C"]
    if cap is None:
        return pool
    if not np.isfinite(cap) or cap < 0:
        raise ValueError("max_dT90_99_C must be null or nonnegative")
    sd = strict_csv(out/"scheil_descriptors.csv")
    valid = sd.loc[sd.scheil_status.eq("ok") & sd.dT90_99_C.between(0, cap)]
    good = {key(*r) for r in valid[KEYS].to_numpy()}
    filtered = pool.loc[[key(*r) in good for r in pool[KEYS].to_numpy()]].copy()
    if filtered.empty:
        raise ValueError("No candidates with successful Scheil results below configured dT limit")
    return filtered


def write_template(candidates, cfg, path):
    if path.exists():
        raise ValueError(f"Template exists: choose a new --template-file ({path})")
    df = candidates[KEYS].copy()
    df.insert(0, "sample_id", [f"P{i:03d}" for i in range(1, len(df)+1)])
    for c in ["icorr_mean_A_cm2", "icorr_sd_A_cm2", "Epit_mean_V_AgAgCl", "Epit_sd_V", "Rct_mean_ohm_cm2", "Rct_sd_ohm_cm2", "Ecorr_mean_V_AgAgCl", "n_replicates"]:
        df[c] = np.nan
    df["experiment_context"] = cfg["experiment_context"]
    df["notes"] = ""
    save_csv(df, path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["check", "equilibrium", "initial", "scheil", "template", "recommend"])
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("settings.json"))
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--limit", type=int, help="Maximum new TC points in this invocation")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--n", type=int, default=10, help="Number of initial experimental compositions")
    parser.add_argument("--candidates", type=Path)
    parser.add_argument("--experiments", type=Path)
    parser.add_argument("--template-file", type=Path)
    args = parser.parse_args(argv)
    cfg = load_settings(args.config)
    out = args.out.resolve()
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    if args.stage == "check":
        print(f"Grid: {len(grid(cfg))} points; equilibrium={cfg['temperature_C']} C; DB={cfg['database']}")
        from tc_python import TCPython
        from tc_backend import build_system, equilibrium, scheil
        with TCPython() as session:
            system = build_system(session, cfg)
            ce, cr = cfg["ce_grid"][0], cfg["cr_grid"][0]
            print("Equilibrium:", equilibrium(system, cfg, ce, cr))
            print("Scheil:", scheil(system, cfg, ce, cr)[0])
        return
    checkpoint = Checkpoint(out, cfg)
    checkpoint.close()
    if args.stage == "equilibrium":
        run_tc("equilibrium", grid(cfg), cfg, out, args.limit, args.retry_failed)
        return
    pool = read_pool(out, cfg)
    if args.stage == "initial":
        save_csv(diverse(manufacturing_pool(pool, out, cfg), args.n, cfg), out/"initial_candidates.csv")
    elif args.stage == "scheil":
        # Explicit --candidates bounds cost; omission intentionally processes whole pool.
        candidates = checked_candidates(args.candidates, pool) if args.candidates else pool
        run_tc("scheil", candidates, cfg, out, args.limit, args.retry_failed)
    elif args.stage == "template":
        source = args.candidates or out/"initial_candidates.csv"
        candidates = checked_candidates(source, manufacturing_pool(pool, out, cfg))
        write_template(candidates, cfg, args.template_file or out/"sp240_results.csv")
    elif args.stage == "recommend":
        from learning import recommend
        filtered = manufacturing_pool(pool, out, cfg)
        selected = recommend(filtered, args.experiments or out/"sp240_results.csv", cfg, out)
        print(selected[["selection_type", *KEYS]].to_string(index=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
