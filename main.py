"""
Hierarchical Constrained Ensemble (HCE) Forecaster for GTA VI Exophase Trophy Earn Rates.
Optimized for Pinball Loss on Logit Scale with Bayesian Launch Drift, Dual-Trend Regression,
and Monotonic Story DAG Projections.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.64485, -0.67449, 0.0, 0.67449, 1.64485]
EPOCH = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)


def clamp(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    p = clamp(pct)
    return math.log(p / (100.0 - p))


def inv_logit(lg: float) -> float:
    if lg > 18.0:
        return 99.99
    if lg < -18.0:
        return 0.01
    return 100.0 / (1.0 + math.exp(-lg))


def round4(x: float) -> float | int:
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    val = whole / 10000.0
    return int(val) if val.is_integer() else val


def parse_iso(ts_str: str) -> datetime:
    return datetime.fromisoformat(ts_str.replace("Z", "+00:00"))


def linear_regression(xs: list[float], ys: list[float]) -> tuple[float, float, float]:
    n = len(xs)
    if n < 2:
        return (ys[0] if ys else 0.0), 0.0, 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    var_x = sum((x - mx) ** 2 for x in xs)
    if var_x <= 1e-12:
        return my, 0.0, 0.0
    cov_xy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = cov_xy / var_x
    intercept = my - slope * mx
    res_var = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys)) / max(1, n - 2)
    return intercept, slope, math.sqrt(res_var)


def pava_descending(arr: list[float]) -> list[float]:
    """Pool Adjacent Violators Algorithm for non-increasing monotonic sequence: arr[0] >= arr[1] >= ... >= arr[n-1]."""
    blocks = [[val, 1] for val in arr]
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][0] < blocks[i + 1][0]:
            sum_val = blocks[i][0] * blocks[i + 1][1] + blocks[i + 1][0] * blocks[i + 1][1]
            tot_w = blocks[i][1] + blocks[i + 1][1]
            blocks[i] = [sum_val / tot_w, tot_w]
            blocks.pop(i + 1)
            if i > 0:
                i -= 1
        else:
            i += 1
    out = []
    for val, w in blocks:
        out.extend([val] * w)
    return out


def forecast_case(case: dict, name: str = "") -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"Invalid levels {case.get('levels')}")
    if not case["history"]:
        raise ValueError("No history provided")

    last = case["history"][-1]
    horizon_days = case["horizon_minutes"] / 1440.0
    sqrt_d = math.sqrt(horizon_days)

    target_dt = parse_iso(case["target"])
    target_hours = (target_dt - EPOCH).total_seconds() / 3600.0
    u_target = math.log(1.0 + max(0.0, target_hours))

    history_reads = []
    for h in case["history"]:
        try:
            dt = parse_iso(h["at"])
            hrs = (dt - EPOCH).total_seconds() / 3600.0
            history_reads.append((hrs, h.get("rates") or []))
        except Exception:
            continue

    num_trophies = len(case["trophies"])
    id_to_idx = {t["id"]: i for i, t in enumerate(case["trophies"])}
    trophy_rows = []

    for i, trophy in enumerate(case["trophies"]):
        tid = trophy["id"]
        tier = trophy.get("tier", "bronze")
        pts = trophy.get("points", 15)
        secret = trophy.get("secret", False)
        rate = last["rates"][i] if i < len(last["rates"]) else None
        if rate is None:
            rate = 1.0
        base_lg = logit(rate)

        # Collect historical observations for this trophy
        pts_history = []
        for hrs, rates in history_reads:
            if i < len(rates) and rates[i] is not None:
                pts_history.append((hrs, logit(rates[i])))

        # Check for empirical trend across history
        has_trend = False
        if len(pts_history) >= 3:
            first_y = pts_history[0][1]
            last_y = pts_history[-1][1]
            if abs(last_y - first_y) > 0.01:
                has_trend = True

        if has_trend:
            ts = [p[0] for p in pts_history]
            us = [math.log(1.0 + max(0.0, t)) for t in ts]
            ys = [p[1] for p in pts_history]

            a_log, b_log, s_log = linear_regression(us, ys)
            a_lin, b_lin, s_lin = linear_regression(ts, ys)

            # Enforce non-negative progress drift
            b_log = max(0.0, b_log)
            b_lin = max(0.0, b_lin)

            y_log = a_log + b_log * u_target
            y_lin = a_lin + b_lin * target_hours
            mu = 0.65 * y_log + 0.35 * y_lin
            mu = max(base_lg - 0.02, mu)
            sigma = math.sqrt(s_log ** 2 + (0.60 * sqrt_d) ** 2)
        else:
            # Calibrated Bayesian prior for launch night
            if rate >= 98.0:
                v_prior = 0.0
                sigma = 0.05
            elif tid == 1 or tid == 14:  # Platinum or 100% completion
                v_prior = 0.01
                sigma = 0.75
            elif tid == 7:  # Final mission (Gold)
                v_prior = 0.07
                sigma = 0.85
            elif tid == 6:  # Chapter 4
                v_prior = 0.14
                sigma = 0.90
            elif tid == 5:  # Chapter 3
                v_prior = 0.20
                sigma = 0.95
            elif tid == 4:  # Chapter 2
                v_prior = 0.28
                sigma = 0.95
            elif tid == 3:  # Chapter 1
                v_prior = 0.38
                sigma = 1.00
            elif secret and pts <= 15:
                v_prior = 0.25
                sigma = 0.95
            elif pts >= 90:
                v_prior = 0.05
                sigma = 0.80
            elif pts >= 30:
                v_prior = 0.12
                sigma = 0.90
            else:
                v_prior = 0.18
                sigma = 0.95

            mu = base_lg + v_prior * horizon_days
            spread = sigma * sqrt_d

        q_row = [round4(inv_logit(mu + z * spread)) for z in Z]
        for k in range(1, len(q_row)):
            if q_row[k] < q_row[k - 1]:
                q_row[k] = q_row[k - 1]
        trophy_rows.append({"id": tid, "q": q_row})

    # DAG Monotonic Post-Processing on Story Chain: Prologue >= Ch1 >= Ch2 >= Ch3 >= Ch4 >= Final
    story_ids = [2, 3, 4, 5, 6, 7]
    valid_story = [tid for tid in story_ids if tid in id_to_idx]
    if len(valid_story) >= 2:
        for lvl in range(5):
            vals = [trophy_rows[id_to_idx[tid]]["q"][lvl] for tid in valid_story]
            smoothed = pava_descending(vals)
            for tid, sm in zip(valid_story, smoothed):
                trophy_rows[id_to_idx[tid]]["q"][lvl] = round4(sm)

    # Platinum Upper Bound: Platinum <= min(Final Mission, 100% Completion)
    if 1 in id_to_idx:
        plat_idx = id_to_idx[1]
        upper_ids = [tid for tid in [7, 14] if tid in id_to_idx]
        if upper_ids:
            for lvl in range(5):
                min_up = min(trophy_rows[id_to_idx[uid]]["q"][lvl] for uid in upper_ids)
                if trophy_rows[plat_idx]["q"][lvl] > min_up:
                    trophy_rows[plat_idx]["q"][lvl] = min_up

    # Final monotonicity pass per trophy
    for row in trophy_rows:
        for k in range(1, len(row["q"])):
            if row["q"][k] < row["q"][k - 1]:
                row["q"][k] = row["q"][k - 1]

    return {"trophies": trophy_rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        out_file = OUTPUT_DIR / f"{path.stem}.out"
        out_file.write_text(json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
