#!/usr/bin/env python3
"""
Compute energy from node utilization traces using
P(u) = P_idle + (P_peak - P_idle) * u**alpha

Usage examples:
  python3 power_idle_dynamic.py /path/to/output_dir --p-idle 100 --p-peak 250 --alpha 1.5
  python3 power_idle_dynamic.py /path/to/node1_util.csv --time-col time --util-col util --p-idle 50 --p-peak 200

Input can be:
 - a single CSV file with a time column and one or more utilization columns (util values in [0,1])
 - a directory: all .csv files inside are treated as per-node traces

Output:
 - prints per-file (per-node) energy (Joules) and total energy
 - writes results to energy_summary.csv in the input directory or same folder as input file
"""
import argparse
import os
import sys
import glob

try:
    import pandas as pd
    import numpy as np
except Exception as e:
    print("This script requires pandas and numpy. Install with: pip install pandas numpy", file=sys.stderr)
    sys.exit(2)


def clamp01(x):
    return np.minimum(np.maximum(x, 0.0), 1.0)


def energy_from_trace(time, util, p_idle, p_peak, alpha):
    # Ensure numpy arrays
    t = np.asarray(time, dtype=float)
    u = np.asarray(util, dtype=float)
    if t.size < 2:
        return 0.0
    # sort by time just in case
    order = np.argsort(t)
    t = t[order]
    u = u[order]
    u = clamp01(u)
    # segment durations: dt_i = t[i+1] - t[i]
    dt = np.diff(t)
    # use util value at left endpoint for each segment
    u_seg = u[:-1]
    power = p_idle + (p_peak - p_idle) * (u_seg ** alpha)
    energy = np.sum(power * dt)  # Joules if power is Watts and time in seconds
    return float(energy)


def find_csv_files(path):
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*.csv")))
        return files
    elif os.path.isfile(path):
        return [path]
    else:
        raise FileNotFoundError(f"No such file or directory: {path}")


def infer_time_and_util_cols(df, time_col_hint, util_col_hint):
    # time column
    if time_col_hint and time_col_hint in df.columns:
        time_col = time_col_hint
    else:
        # common names
        for c in ("time", "timestamp", "t", "sec", "seconds"):
            if c in df.columns:
                time_col = c
                break
        else:
            # fallback to first numeric column
            numeric = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
            if not numeric:
                raise ValueError("No numeric columns found to use as time")
            time_col = numeric[0]

    # utilization columns
    util_cols = []
    if util_col_hint:
        if util_col_hint in df.columns:
            util_cols = [util_col_hint]
        else:
            # allow comma-separated hints
            for c in util_col_hint.split(","):
                c = c.strip()
                if c in df.columns:
                    util_cols.append(c)
    if not util_cols:
        # heuristics: columns containing "util", "usage", "cpu", "load"
        for c in df.columns:
            if c == time_col:
                continue
            lname = c.lower()
            if any(k in lname for k in ("util", "usage", "cpu", "load", "occup")):
                util_cols.append(c)
        if not util_cols:
            # fallback: all numeric columns except time_col
            util_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number) and c != time_col]

    if not util_cols:
        raise ValueError("Unable to infer utilization columns from file. Please provide --util-col.")

    return time_col, util_cols


def process_file(path, args):
    df = pd.read_csv(path)
    time_col, util_cols = infer_time_and_util_cols(df, args.time_col, args.util_col)
    results = []
    for col in util_cols:
        energy = energy_from_trace(df[time_col].values, df[col].values, args.p_idle, args.p_peak, args.alpha)
        results.append((col, energy))
    return results, time_col, util_cols


def main():
    parser = argparse.ArgumentParser(description="Energy calculation: P(u)=P_idle+(P_peak-P_idle)*u^alpha")
    parser.add_argument("input", help="CSV file or directory containing CSV traces")
    parser.add_argument("--p-idle", type=float, required=True, help="Idle power in Watts (P_idle)")
    parser.add_argument("--p-peak", type=float, required=True, help="Peak power in Watts (P_peak)")
    parser.add_argument("--alpha", type=float, default=1.0, help="Curvature alpha (>=1)")
    parser.add_argument("--time-col", type=str, default=None, help="Name of time column (seconds)")
    parser.add_argument("--util-col", type=str, default=None, help="Name(s) of utilization column(s). Comma-separated or leave to auto-detect")
    parser.add_argument("--out", type=str, default=None, help="Output CSV summary path")
    args = parser.parse_args()

    try:
        files = find_csv_files(args.input)
    except FileNotFoundError as e:
        print(str(e), file=sys.stderr)
        sys.exit(2)

    summary_rows = []
    total_energy = 0.0
    for f in files:
        try:
            results, time_col, util_cols = process_file(f, args)
        except Exception as e:
            print(f"Skipping {os.path.basename(f)}: {e}", file=sys.stderr)
            continue
        # if multiple util cols, aggregate energies and also report per-column
        file_energy = 0.0
        for col, energy in results:
            summary_rows.append({
                "file": os.path.basename(f),
                "column": col,
                "energy_joule": energy,
                "p_idle_W": args.p_idle,
                "p_peak_W": args.p_peak,
                "alpha": args.alpha
            })
            file_energy += energy
        total_energy += file_energy
        print(f"{os.path.basename(f)}: energy = {file_energy:.3f} J  (columns: {', '.join([c for c,_ in results])})")

    print(f"TOTAL energy (all files) = {total_energy:.3f} J")

    # write CSV
    out_path = args.out
    if out_path is None:
        if os.path.isdir(args.input):
            out_path = os.path.join(args.input, "energy_summary.csv")
        else:
            out_path = os.path.join(os.path.dirname(os.path.abspath(args.input)), "energy_summary.csv")
    try:
        pd.DataFrame(summary_rows).to_csv(out_path, index=False)
        print(f"Summary written to {out_path}")
    except Exception as e:
        print(f"Unable to write summary CSV: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()