"""Real TC-Python adapter. No synthetic fallback and no forced matrix phase."""
import json
import warnings
import numpy as np
from core import comp, scheil_metrics


def build_system(session, cfg):
    elements = ["FE", *cfg["base_wt"], "CR", "CE"]
    system = session.select_database_and_elements(cfg["database"], elements).get_system()
    available = {p.split("#")[0] for p in system.get_phases_in_system()}
    if cfg["matrix_phase"] not in available:
        raise ValueError(f"Matrix phase {cfg['matrix_phase']} unavailable in selected system")
    missing = set(cfg["harmful_phases"])-available
    if missing:
        warnings.warn(f"Harmful phases absent from this system (NOT evidence of physical absence): {sorted(missing)}. Check database phase names and coverage.")
    return system


def equilibrium(system, cfg, ce, cr):
    from tc_python import ThermodynamicQuantity as Q
    calc = (system.with_single_equilibrium_calculation()
            .set_condition(Q.temperature(), cfg["temperature_C"] + 273.15)
            .set_condition(Q.pressure(), cfg["pressure_Pa"]))
    for element, wt in comp(cfg, ce, cr).items():
        calc.set_condition(Q.mass_fraction_of_a_component(element), wt / 100.)
    result = calc.calculate()
    phases = list(result.get_stable_phases())
    # Query exact composition-set names (#1/#2), THEN aggregate base names.
    fractions = {p: float(result.get_value_of(Q.mole_fraction_of_a_phase(p))) for p in phases}
    if not np.isfinite(list(fractions.values())).all() or abs(sum(fractions.values())-1.) > 1e-4 or min(fractions.values()) < -1e-8:
        raise ValueError("Invalid phase fractions: do not treat query failures as zero")
    matrix = [p for p in phases if p.split("#")[0] == cfg["matrix_phase"]]
    matrix_fraction = sum(fractions[p] for p in matrix)
    matrix_cr = np.nan
    if matrix:
        mass = np.array([result.get_value_of(Q.mass_fraction_of_a_phase(p)) for p in matrix], float)
        crs = np.array([result.get_value_of(Q.composition_of_phase_as_weight_fraction(p, "CR")) for p in matrix], float)
        if not np.isfinite(mass).all() or not np.isfinite(crs).all() or mass.sum() <= 0:
            raise ValueError("Invalid matrix composition query")
        matrix_cr = float(100*np.dot(mass, crs)/mass.sum())
    harmful = sum(f for p, f in fractions.items() if p.split("#")[0] in cfg["harmful_phases"])
    return dict(matrix_phase_fraction=matrix_fraction, matrix_Cr_wt=matrix_cr,
                harmful_phase_fraction=harmful, stable_phases=";".join(phases),
                phase_fractions_json=json.dumps(fractions), calculation_status="ok", calculation_error="")


def scheil(system, cfg, ce, cr):
    from tc_python import ScheilQuantity as Q, CompositionUnit
    calc = (system.with_scheil_calculation()
            .set_composition_unit(CompositionUnit.MASS_PERCENT)
            .set_start_temperature(cfg["scheil_start_K"]))
    for element, wt in comp(cfg, ce, cr).items():
        calc.set_composition(element, wt)
    result = calc.calculate()
    t, fs = result.get_values_of(Q.temperature(), Q.mole_fraction_of_all_solid_phases())
    metrics = scheil_metrics(t, fs)
    metrics.update(scheil_status="ok", scheil_error="")
    curves = {"solid_fraction": {"temperature_K": list(t), "solid_mole_fraction": list(fs)}}
    # Each query can return a different temperature grid: preserve it separately.
    segregation_errors = []
    for element in ("CE", "CR"):
        try:
            temp, fraction = result.get_values_of(Q.temperature(), Q.composition_of_phase_as_weight_fraction("LIQUID", element))
            temp, fraction = np.asarray(temp, float), np.asarray(fraction, float)
            valid = np.isfinite(temp) & np.isfinite(fraction)
            temp, fraction = temp[valid], fraction[valid]
            if not len(temp):
                raise ValueError("No valid residual-liquid composition values")
            curves[f"liquid_{element}"] = {"temperature_K": temp.tolist(), f"liquid_{element}_wt": (fraction*100).tolist()}
            metrics[f"liquid_{element}_last_returned_wt"] = float(fraction[np.argmin(temp)]*100)
        except Exception as exc:
            segregation_errors.append(f"{element}: {exc}")
    metrics["segregation_status"] = "partial_or_failed" if segregation_errors else "ok"
    metrics["segregation_error"] = "; ".join(segregation_errors)
    return metrics, curves
