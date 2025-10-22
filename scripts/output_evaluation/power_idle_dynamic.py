#!/usr/bin/env python3
"""
Energy from utilization traces (Baustein 1)
P(u) = P_idle + (P_peak - P_idle) * u**alpha
E = ∫ P(u(t)) dt

Sources supported (auto-detected or via --prefer):
 - cpu_utilization.csv  (preferred)
 - gpu_utilization.csv
 - node_utilization.csv (derive util from State: free -> 0 else 1)

Ignored for Baustein 1: event.csv, job_statistics.csv, network_activity.csv, pfs_utilization.csv

Outputs:
 - Console: per-node energy + TOTAL
 - CSV: energy_summary.csv with columns:
   source,file,node,energy_joule,mean_util,p_idle_W,p_peak_W,alpha
"""

import argparse
import os
import sys
import glob

try:
    import pandas as pd
    import numpy as np
except Exception:
    print("This script requires pandas and numpy. Install with: pip install pandas numpy", file=sys.stderr)
    sys.exit(2)


# ---------- math core ----------

def clamp01(x):
    return np.minimum(np.maximum(x, 0.0), 1.0)


def energy_from_trace(time, util, p_idle, p_peak, alpha, time_scale=1.0):
    t = np.asarray(time, dtype=float)  * float(time_scale)
    u = np.asarray(util, dtype=float)
    if t.size < 2:
        return 0.0
    order = np.argsort(t)
    t = t[order]
    u = clamp01(u[order])
    dt = np.diff(t)
    if np.any(dt < 0):
        # After sort this should not happen; guard anyway
        dt = np.maximum(dt, 0.0)
    u_seg = u[:-1]  # left Riemann
    power = p_idle + (p_peak - p_idle) * (u_seg ** alpha)
    energy = float(np.sum(power * dt))  # W*s = J
    return energy


# ---------- source readers ----------

def looks_like_util(series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return False
    min_v, max_v = float(s.min()), float(s.max())
    # accept either [0,1] or [0,100] (percentage)
    return (0.0 <= min_v <= 1.0 and 0.0 <= max_v <= 1.0) or (0.0 <= min_v <= 100.0 and 0.0 <= max_v <= 100.0)


def normalize_util(series):
    s = pd.to_numeric(series, errors="coerce")
    # scale 0..100 to 0..1 if needed
    if s.max(skipna=True) > 1.001:
        s = s / 100.0
    # clamp just in case
    s = s.clip(lower=0.0, upper=1.0)
    return s


def read_cpu_or_gpu_util(path, util_candidates=None):
    """
    Unterstützt CPU/GPU-Utilization im Wide-Format (Time + <Node_0>, <Node_1>, ...)
    und Long-Format (Time, Node, Util).
    Gibt ein Dict zurück: node -> (time ndarray, util ndarray).
    """
    df = pd.read_csv(path)
    cols = list(df.columns)
    cols_lower = {c.lower(): c for c in cols}

    # --- Time-Spalte ermitteln ---
    time_col = None
    for c in ("time", "timestamp", "t", "sec", "seconds"):
        if c in cols_lower:
            time_col = cols_lower[c]
            break
    if time_col is None:
        # Fallback: erste numerische Spalte als Zeit
        numeric = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
        if not numeric:
            raise ValueError("No numeric column available to use as time")
        time_col = numeric[0]

    # --- Prüfen, ob Long-Format existiert (Time, Node, Util) ---
    node_col = None
    for c in ("node", "host", "hostname", "nid", "id"):
        if c in cols_lower:
            node_col = cols_lower[c]
            break

    # Kandidaten für Util-Spaltennamen (nur für Long-Format relevant)
    util_candidates = util_candidates or [
        "util", "utilization", "cpu_util", "cpu utilization", "gpu_util", "gpu utilization",
        "usage", "value", "load"
    ]
    util_col = None
    for cand in util_candidates:
        if cand in cols_lower:
            util_col = cols_lower[cand]
            break

    # --- Fall A: Wide-Format (Time + viele Node-Spalten) ---
    # Heuristik: Wenn es KEIN node_col gibt, interpretieren wir alle
    # nicht-Time-Spalten als Nodes mit Util-Werten.
    if node_col is None:
        node_columns = [c for c in cols if c != time_col]
        # Sicherheit: mind. 1 Node-Spalte
        if not node_columns:
            raise ValueError("Wide-format detection failed: no node columns found")
        # Werte normalisieren auf [0,1] (falls 0..100)
        traces = {}
        # sortieren nach Zeit
        df = df.sort_values(time_col)
        t = df[time_col].to_numpy(dtype=float)
        for nc in node_columns:
            s = pd.to_numeric(df[nc], errors="coerce")
            # 0..100 → 0..1
            if s.max(skipna=True) > 1.001:
                s = s / 100.0
            s = s.clip(lower=0.0, upper=1.0)
            traces[str(nc)] = (t, s.to_numpy(dtype=float))
        return traces

    # --- Fall B: Long-Format (Time, Node, Util) ---
    if util_col is None:
        # falls kein Namenskandidat passt, suche Spalte, die wie Util aussieht
        for c in cols:
            if c in (time_col, node_col):
                continue
            # einfache Range-Heuristik
            s = pd.to_numeric(df[c], errors="coerce").dropna()
            if not s.empty:
                min_v, max_v = float(s.min()), float(s.max())
                if (0.0 <= min_v <= 1.0 and 0.0 <= max_v <= 1.0) or (0.0 <= min_v <= 100.0 and 0.0 <= max_v <= 100.0):
                    util_col = c
                    break
    if util_col is None:
        raise ValueError("No utilization column found for long-format data")

    # Auf benötigte Spalten reduzieren und normalisieren
    df = df[[time_col, node_col, util_col]].copy()
    df.rename(columns={time_col: "Time", node_col: "Node", util_col: "Util"}, inplace=True)
    # 0..100 → 0..1
    if df["Util"].max(skipna=True) > 1.001:
        df["Util"] = df["Util"] / 100.0
    df["Util"] = df["Util"].clip(lower=0.0, upper=1.0)

    traces = {}
    for node, g in df.groupby("Node"):
        g = g.sort_values("Time")
        traces[str(node)] = (g["Time"].to_numpy(dtype=float), g["Util"].to_numpy(dtype=float))
    return traces



def read_node_state_util(path):
    """
    Read node_utilization.csv with columns like: Time, Node, State, ...
    Util is derived from State: free -> 0 else -> 1
    """
    df = pd.read_csv(path)
    required = {"Time", "Node", "State"}
    if not required.issubset(df.columns):
        raise ValueError("node_utilization.csv missing required columns Time/Node/State")
    util = (df["State"].astype(str).str.lower() != "free").astype(float)
    df = df[["Time", "Node"]].copy().assign(Util=util)

    traces = {}
    for node, g in df.groupby("Node"):
        g = g.sort_values("Time")
        traces[str(node)] = (g["Time"].to_numpy(dtype=float), g["Util"].to_numpy(dtype=float))
    return traces


# ---------- discovery & filtering ----------

def discover_source_files(dir_path):
    # Return dict of known sources present in directory
    present = {os.path.basename(p).lower(): p for p in glob.glob(os.path.join(dir_path, "*.csv"))}
    sources = {
        "cpu": present.get("cpu_utilization.csv"),
        "gpu": present.get("gpu_utilization.csv"),
        "node": present.get("node_utilization.csv"),
        # we intentionally ignore others for Baustein 1
    }
    return sources


def pick_source(sources, prefer):
    order = []
    if prefer:
        order = [prefer]
    # default priority: cpu -> gpu -> node
    for s in ("cpu", "gpu", "node"):
        if s not in order:
            order.append(s)
    for s in order:
        if sources.get(s):
            return s, sources[s]
    raise FileNotFoundError("No suitable utilization source found (cpu/gpu/node).")


# ---------- main pipeline ----------

def main():
    ap = argparse.ArgumentParser(description="Energy calculation (Baustein 1): P(u)=P_idle+(P_peak-P_idle)*u^alpha")
    ap.add_argument("input", help="CSV file or directory with ElastiSim outputs")
    ap.add_argument("--p-idle", type=float, required=True, help="Idle power in Watts")
    ap.add_argument("--p-peak", type=float, required=True, help="Peak power in Watts")
    ap.add_argument("--alpha", type=float, default=1.0, help="Curvature alpha (>=1)")
    ap.add_argument("--prefer", choices=["cpu", "gpu", "node"], default=None,
                    help="Prefer a specific source file if multiple exist in directory")
    ap.add_argument("--out", type=str, default=None, help="Output CSV summary path")
    ap.add_argument("--time-scale", type=float, default=1.0,
                    help="Zeit-Skalierungsfaktor für die Time-Spalte (z.B. 0.001 für Millisekunden → Sekunden)")
    args = ap.parse_args()

    path = args.input
    summary_rows = []
    total_energy = 0.0

    if os.path.isdir(path):
        sources = discover_source_files(path)
        source_key, source_file = pick_source(sources, args.prefer)
        print(f"Selected source: {source_key} -> {os.path.basename(source_file)}")
        if source_key == "node":
            traces = read_node_state_util(source_file)
        else:
            traces = read_cpu_or_gpu_util(source_file)
        source_label = source_key
        in_file_for_csv = os.path.basename(source_file)
    elif os.path.isfile(path):
        base = os.path.basename(path).lower()
        if base == "node_utilization.csv":
            traces = read_node_state_util(path)
            source_label = "node"
        elif base in ("cpu_utilization.csv", "gpu_utilization.csv"):
            traces = read_cpu_or_gpu_util(path)
            source_label = "cpu" if "cpu_" in base else "gpu"
        else:
            # last resort: try generic CPU/GPU reader, then node-state
            try:
                traces = read_cpu_or_gpu_util(path)
                source_label = "generic"
            except Exception:
                traces = read_node_state_util(path)
                source_label = "node"
        in_file_for_csv = os.path.basename(path)
    else:
        print(f"No such file or directory: {path}", file=sys.stderr)
        sys.exit(2)

    # compute per-node energy
    for node, (t, u) in traces.items():
        E = energy_from_trace(t, u, args.p_idle, args.p_peak, args.alpha, args.time_scale)
        mean_u = float(np.nanmean(u)) if len(u) else 0.0
        total_energy += E
        print(f"{in_file_for_csv} | node={node}: energy = {E:.3f} J  (mean util={mean_u:.3f})")
        summary_rows.append({
            "source": source_label,
            "file": in_file_for_csv,
            "node": node,
            "energy_joule": E,
            "mean_util": mean_u,
            "p_idle_W": args.p_idle,
            "p_peak_W": args.p_peak,
            "alpha": args.alpha,
        })


    any_node = next(iter(traces.values()))
    t_any = any_node[0]
    raw_T = float(t_any[-1] - t_any[0]) if len(t_any) >= 2 else 0.0
    T = raw_T * args.time_scale  # <-- Zeit korrekt skalieren

    avg_cluster_power_W = total_energy / T if T > 0 else float("nan")

    print(f"Simulated duration T = {T:.3f} s")
    print(f"Average cluster power = {avg_cluster_power_W:.2f} W")
    print(f"TOTAL energy = {total_energy:.3f} J  (~{total_energy/3.6e6:.3f} kWh)")


    # write CSV
    out_path = args.out
    if out_path is None:
        if os.path.isdir(path):
            out_path = os.path.join(path, "energy_summary.csv")
        else:
            out_dir = os.path.dirname(os.path.abspath(path)) or "."
            out_path = os.path.join(out_dir, "energy_summary.csv")
    try:
        pd.DataFrame(summary_rows).to_csv(out_path, index=False)
        print(f"Summary written to {out_path}")
    except Exception as e:
        print(f"Unable to write summary CSV: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
