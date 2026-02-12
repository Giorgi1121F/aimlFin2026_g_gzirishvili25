#!/usr/bin/env python3
"""
code.py — DDoS interval detection from event logs using regression + visualizations

Usage examples:
  python code.py --log task_3/events.log --out task_3/out
  python code.py --log task_3/events.csv --out task_3/out --time-col timestamp
  python code.py --log task_3/events.jsonl --out task_3/out --time-col time --metric-col requests

Outputs (in --out):
  - ddos_intervals.csv
  - timeseries.csv
  - regression_plot.png
  - residuals_plot.png
  - rate_plot.png
  - run_summary.txt
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from datetime import timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# --------------------------
# Parsing helpers
# --------------------------

TS_REGEXES = [
    # 2026-02-12 12:34:56, 2026-02-12T12:34:56, 2026-02-12T12:34:56Z, 2026-02-12T12:34:56.123Z
    r"(?P<ts>\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?Z?)",
    # 12/Feb/2026:12:34:56 +0000 (common web logs)
    r"(?P<ts>\d{2}/[A-Za-z]{3}/\d{4}:\d{2}:\d{2}:\d{2}(?: [+\-]\d{4})?)",
    # epoch seconds / ms: 1700000000 or 1700000000000
    r"(?P<ts>\b\d{10}\b|\b\d{13}\b)",
]

MONTH_MAP = {
    "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4,
    "May": 5, "Jun": 6, "Jul": 7, "Aug": 8,
    "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12
}


def try_parse_timestamp(value: str) -> Optional[pd.Timestamp]:
    if value is None:
        return None
    s = str(value).strip()

    # epoch
    if re.fullmatch(r"\d{10}", s):
        try:
            return pd.to_datetime(int(s), unit="s", utc=True)
        except Exception:
            return None
    if re.fullmatch(r"\d{13}", s):
        try:
            return pd.to_datetime(int(s), unit="ms", utc=True)
        except Exception:
            return None

    # Apache-like 12/Feb/2026:12:34:56 +0000
    m = re.fullmatch(r"(\d{2})/([A-Za-z]{3})/(\d{4}):(\d{2}):(\d{2}):(\d{2})(?: ([+\-]\d{4}))?", s)
    if m:
        dd, mon, yyyy, hh, mm, ss, off = m.groups()
        mon_i = MONTH_MAP.get(mon, 1)
        dt = pd.Timestamp(
            year=int(yyyy),
            month=int(mon_i),
            day=int(dd),
            hour=int(hh),
            minute=int(mm),
            second=int(ss),
            tz="UTC",
        )
        # If offset exists, convert to UTC
        if off:
            sign = 1 if off[0] == "+" else -1
            off_h = int(off[1:3])
            off_m = int(off[3:5])
            delta = pd.Timedelta(hours=sign * off_h, minutes=sign * off_m)
            dt = (dt - delta).tz_convert("UTC")
        return dt

    # ISO-ish
    try:
        ts = pd.to_datetime(s, utc=True, errors="raise")
        if isinstance(ts, pd.Timestamp) and ts.tz is None:
            ts = ts.tz_localize("UTC")
        return ts
    except Exception:
        return None


def extract_timestamp_from_line(line: str) -> Optional[pd.Timestamp]:
    for rx in TS_REGEXES:
        m = re.search(rx, line)
        if not m:
            continue
        raw = m.group("ts")
        ts = try_parse_timestamp(raw)
        if ts is not None:
            return ts
    return None


def sniff_format(path: str) -> str:
    ext = os.path.splitext(path.lower())[1]
    if ext in [".jsonl", ".ndjson"]:
        return "jsonl"
    if ext in [".json"]:
        return "json"
    if ext in [".csv", ".tsv"]:
        return "csv"
    return "text"


def read_jsonl(path: str) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    rows.append(obj)
            except Exception:
                continue
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def read_json(path: str) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        try:
            obj = json.load(f)
        except Exception:
            return pd.DataFrame()
    if isinstance(obj, list):
        return pd.DataFrame(obj)
    if isinstance(obj, dict):
        # If dict has a list field, try to find it
        for k, v in obj.items():
            if isinstance(v, list):
                return pd.DataFrame(v)
        return pd.DataFrame([obj])
    return pd.DataFrame()


def read_csv(path: str) -> pd.DataFrame:
    # Try common separators
    for sep in [",", "\t", ";", "|"]:
        try:
            df = pd.read_csv(path, sep=sep)
            if df.shape[1] >= 2:
                return df
        except Exception:
            pass
    # Fallback
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def read_text_lines(path: str, max_lines: int = 2_000_000) -> pd.DataFrame:
    # Build minimal DataFrame with timestamp + raw line
    ts_list = []
    line_list = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if i >= max_lines:
                break
            line = line.rstrip("\n")
            ts = extract_timestamp_from_line(line)
            if ts is None:
                continue
            ts_list.append(ts)
            line_list.append(line)
    if not ts_list:
        return pd.DataFrame()
    return pd.DataFrame({"timestamp": ts_list, "raw": line_list})


def guess_time_column(df: pd.DataFrame) -> Optional[str]:
    if df.empty:
        return None
    candidates = []
    for col in df.columns:
        c = str(col).lower()
        if any(k in c for k in ["time", "timestamp", "date", "datetime", "@timestamp", "ts"]):
            candidates.append(col)

    # Try candidates first
    for col in candidates:
        sample = df[col].dropna().astype(str).head(50).tolist()
        ok = sum(1 for v in sample if try_parse_timestamp(v) is not None)
        if ok >= max(5, int(0.6 * len(sample))):
            return col

    # Else brute-force: find a column that parses well
    best_col = None
    best_ok = 0
    for col in df.columns:
        sample = df[col].dropna().astype(str).head(50).tolist()
        if not sample:
            continue
        ok = sum(1 for v in sample if try_parse_timestamp(v) is not None)
        if ok > best_ok:
            best_ok = ok
            best_col = col
    if best_ok >= 5:
        return best_col
    return None


def to_timestamp_series(df: pd.DataFrame, time_col: str) -> pd.Series:
    s = df[time_col].astype(str)
    parsed = s.map(try_parse_timestamp)
    parsed = pd.to_datetime(parsed, utc=True, errors="coerce")
    return parsed


# --------------------------
# DDoS detection logic
# --------------------------

@dataclass
class DetectionConfig:
    freq: str = "1S"                 # resample frequency
    train_quantile: float = 0.90     # exclude top spikes from baseline fit
    z_threshold: float = 3.0         # residual z-score threshold
    ratio_threshold: float = 2.0     # actual/predicted ratio threshold
    min_interval_seconds: int = 5    # minimum interval duration to keep
    merge_gap_seconds: int = 2       # merge anomalies separated by <= this gap


def build_rate_series(ts: pd.Series, freq: str) -> pd.Series:
    ts = ts.dropna().sort_values()
    if ts.empty:
        return pd.Series(dtype=float)
    idx = pd.DatetimeIndex(ts)
    # count events per bin
    s = pd.Series(1, index=idx).resample(freq).sum().fillna(0.0)
    s.name = "event_count"
    return s


def fit_regression_baseline(rate: pd.Series, train_quantile: float) -> Tuple[np.ndarray, np.ndarray, LinearRegression]:
    """
    Fit baseline with LinearRegression: y ~ time_index
    Exclude extreme points above quantile to avoid fitting to the attack spikes.
    """
    if rate.empty:
        raise ValueError("Empty rate series.")

    y = rate.values.astype(float)
    x = np.arange(len(rate)).reshape(-1, 1).astype(float)

    cutoff = np.quantile(y, train_quantile)
    mask = y <= cutoff

    # If too few points remain, relax
    if mask.sum() < max(30, int(0.3 * len(y))):
        mask = np.ones_like(y, dtype=bool)

    model = LinearRegression()
    model.fit(x[mask], y[mask])

    y_pred = model.predict(x)
    residuals = y - y_pred
    return y_pred, residuals, model


def find_anomalies(rate: pd.Series, y_pred: np.ndarray, residuals: np.ndarray, cfg: DetectionConfig) -> pd.Series:
    # z-score on residuals (robust-ish)
    resid_std = np.std(residuals) + 1e-9
    z = residuals / resid_std

    pred_safe = np.maximum(y_pred, 1e-6)
    ratio = rate.values / pred_safe

    is_anom = (z >= cfg.z_threshold) | (ratio >= cfg.ratio_threshold)
    return pd.Series(is_anom, index=rate.index, name="is_anomaly")


def merge_intervals(flags: pd.Series, cfg: DetectionConfig) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """
    flags: boolean series indexed by time bins.
    """
    if flags.empty:
        return []

    times = flags.index
    vals = flags.values.astype(bool)

    intervals: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    start = None
    last_true_time = None

    bin_seconds = pd.to_timedelta(flags.index.freq or pd.Timedelta(cfg.freq)).total_seconds()
    gap_bins = int(np.ceil(cfg.merge_gap_seconds / max(bin_seconds, 1e-9)))

    for t, v in zip(times, vals):
        if v:
            if start is None:
                start = t
            last_true_time = t
        else:
            if start is not None and last_true_time is not None:
                # allow small gaps
                # check if we are within merge gap; if yes, keep interval open
                # (we implement by delaying closing until gap exceeds threshold)
                pass

    # A simpler grouping with run-length encoding + merge gap
    true_idx = np.where(vals)[0]
    if true_idx.size == 0:
        return []

    # group indices with merge gap
    groups = []
    g_start = true_idx[0]
    g_prev = true_idx[0]
    for i in true_idx[1:]:
        if i - g_prev <= gap_bins:
            g_prev = i
        else:
            groups.append((g_start, g_prev))
            g_start = i
            g_prev = i
    groups.append((g_start, g_prev))

    min_bins = int(np.ceil(cfg.min_interval_seconds / max(bin_seconds, 1e-9)))
    for a, b in groups:
        if (b - a + 1) >= max(1, min_bins):
            start_t = times[a]
            end_t = times[b] + pd.to_timedelta(cfg.freq)  # end is exclusive-ish
            intervals.append((start_t, end_t))

    return intervals


# --------------------------
# Plots
# --------------------------

def plot_rate(rate: pd.Series, out_path: str) -> None:
    plt.figure(figsize=(10, 4))
    plt.plot(rate.index, rate.values)
    plt.title("Event rate over time")
    plt.xlabel("Time (UTC)")
    plt.ylabel("Events per bin")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_regression(rate: pd.Series, y_pred: np.ndarray, intervals: List[Tuple[pd.Timestamp, pd.Timestamp]], out_path: str) -> None:
    plt.figure(figsize=(10, 4))
    plt.plot(rate.index, rate.values, label="Actual rate")
    plt.plot(rate.index, y_pred, label="Regression baseline")
    for (a, b) in intervals:
        plt.axvspan(a, b, alpha=0.2)
    plt.title("Regression baseline vs actual (shaded = detected intervals)")
    plt.xlabel("Time (UTC)")
    plt.ylabel("Events per bin")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_residuals(rate: pd.Series, residuals: np.ndarray, out_path: str) -> None:
    plt.figure(figsize=(10, 4))
    plt.plot(rate.index, residuals)
    plt.axhline(0, linestyle="--")
    plt.title("Residuals (actual - predicted)")
    plt.xlabel("Time (UTC)")
    plt.ylabel("Residual")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# --------------------------
# Main
# --------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, help="Path to provided event log file (csv/json/jsonl/txt).")
    ap.add_argument("--out", required=True, help="Output folder (e.g., task_3/out).")
    ap.add_argument("--time-col", default=None, help="Time column name (for structured logs). If not set, auto-detect.")
    ap.add_argument("--freq", default="1S", help="Resample frequency (e.g., 1S, 5S, 1Min). Default: 1S")
    ap.add_argument("--train-quantile", type=float, default=0.90, help="Exclude top quantile spikes from baseline fit.")
    ap.add_argument("--z-threshold", type=float, default=3.0, help="Residual z-score threshold for anomalies.")
    ap.add_argument("--ratio-threshold", type=float, default=2.0, help="Actual/predicted ratio threshold for anomalies.")
    ap.add_argument("--min-interval-sec", type=int, default=5, help="Minimum DDoS interval length in seconds.")
    ap.add_argument("--merge-gap-sec", type=int, default=2, help="Merge anomalies separated by <= this gap (seconds).")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    fmt = sniff_format(args.log)
    if fmt == "jsonl":
        df = read_jsonl(args.log)
    elif fmt == "json":
        df = read_json(args.log)
    elif fmt == "csv":
        df = read_csv(args.log)
    else:
        df = read_text_lines(args.log)

    if df.empty:
        raise SystemExit("Could not parse any timestamps/events from the log file. Try specifying --time-col or check the file format.")

    # get timestamp series
    if "timestamp" in df.columns and args.time_col is None and fmt == "text":
        ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    else:
        time_col = args.time_col or guess_time_column(df)
        if not time_col:
            raise SystemExit("Could not detect a time column. Please rerun with --time-col <column_name>.")
        ts = to_timestamp_series(df, time_col)

    ts = ts.dropna()
    if ts.empty:
        raise SystemExit("No valid timestamps after parsing. Please check time format or specify --time-col.")

    cfg = DetectionConfig(
        freq=args.freq,
        train_quantile=args.train_quantile,
        z_threshold=args.z_threshold,
        ratio_threshold=args.ratio_threshold,
        min_interval_seconds=args.min_interval_sec,
        merge_gap_seconds=args.merge_gap_sec,
    )

    rate = build_rate_series(ts, cfg.freq)
    if rate.empty or len(rate) < 20:
        raise SystemExit("Not enough data after resampling. Try larger --freq (e.g., 5S or 1Min).")

    y_pred, residuals, model = fit_regression_baseline(rate, cfg.train_quantile)
    flags = find_anomalies(rate, y_pred, residuals, cfg)
    intervals = merge_intervals(flags, cfg)

    # Save timeseries
    out_ts = pd.DataFrame({
        "time_utc": rate.index.astype(str),
        "event_rate": rate.values,
        "predicted_baseline": y_pred,
        "residual": residuals,
        "is_anomaly": flags.values.astype(int),
    })
    out_ts.to_csv(os.path.join(args.out, "timeseries.csv"), index=False)

    # Save intervals
    out_int = pd.DataFrame(
        [{"start_utc": str(a), "end_utc": str(b), "duration_sec": int((b - a).total_seconds())} for a, b in intervals]
    )
    out_int.to_csv(os.path.join(args.out, "ddos_intervals.csv"), index=False)

    # Plots
    plot_rate(rate, os.path.join(args.out, "rate_plot.png"))
    plot_regression(rate, y_pred, intervals, os.path.join(args.out, "regression_plot.png"))
    plot_residuals(rate, residuals, os.path.join(args.out, "residuals_plot.png"))

    # Summary
    summary_lines = [
        f"Log file: {args.log}",
        f"Parsed events: {len(ts)}",
        f"Time range (UTC): {ts.min()}  ->  {ts.max()}",
        f"Resample freq: {cfg.freq}",
        f"Regression: LinearRegression(y ~ time_index), coef={float(model.coef_[0]):.6f}, intercept={float(model.intercept_):.6f}",
        f"Anomaly thresholds: z>={cfg.z_threshold}, ratio>={cfg.ratio_threshold}",
        f"Detected intervals: {len(intervals)}",
        "",
        "Intervals (UTC):",
    ]
    if intervals:
        for a, b in intervals:
            summary_lines.append(f" - {a}  ->  {b}   ({int((b-a).total_seconds())} sec)")
    else:
        summary_lines.append(" - None detected with current thresholds. Try lowering thresholds or changing --freq.")

    with open(os.path.join(args.out, "run_summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))

    print("\n".join(summary_lines))
    print(f"\nSaved outputs to: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())