from __future__ import annotations

from dataclasses import asdict, dataclass
from math import comb, erfc, sqrt
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


ROOT = Path(__file__).resolve().parent
DEFAULT_LOG_PATH = ROOT / "prediction_log.xlsx"

PROBA_COL = "home_win_proba"
RESULT_COL = "result"
MODEL_COL = "model_name"

PostprocessMethod = Literal["none", "shrink", "logit_intercept", "platt"]


@dataclass(frozen=True)
class PostprocessConfig:
    method: PostprocessMethod = "shrink"
    alpha: float = 0.8
    logit_intercept: float = 0.0
    platt_slope: float = 1.0
    platt_intercept: float = 0.0
    clip_eps: float = 1e-6
    name: str | None = None

    @property
    def label(self) -> str:
        if self.name:
            return self.name
        if self.method == "none":
            return "raw"
        if self.method == "shrink":
            return f"shrink_alpha_{self.alpha:.3g}"
        if self.method == "logit_intercept":
            return f"logit_intercept_{self.logit_intercept:.3g}"
        if self.method == "platt":
            return f"platt_slope_{self.platt_slope:.3g}_intercept_{self.platt_intercept:.3g}"
        return self.method


def clip_proba(proba, eps: float = 1e-6) -> np.ndarray:
    values = pd.to_numeric(pd.Series(proba), errors="coerce").to_numpy(dtype=float)
    return np.clip(values, eps, 1.0 - eps)


def sigmoid(x) -> np.ndarray:
    x_arr = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-x_arr))


def logit(proba, eps: float = 1e-6) -> np.ndarray:
    p = clip_proba(proba, eps=eps)
    return np.log(p / (1.0 - p))


def shrink_proba(proba, alpha: float = 0.8, center: float = 0.5, eps: float = 1e-6) -> np.ndarray:
    if alpha < 0:
        raise ValueError("alpha must be non-negative.")
    p = clip_proba(proba, eps=eps)
    return np.clip(center + alpha * (p - center), eps, 1.0 - eps)


def logit_shift_proba(
    proba,
    intercept: float = 0.0,
    slope: float = 1.0,
    eps: float = 1e-6,
) -> np.ndarray:
    return np.clip(sigmoid(slope * logit(proba, eps=eps) + intercept), eps, 1.0 - eps)


def apply_proba_postprocess(proba, config: PostprocessConfig) -> np.ndarray:
    if config.method == "none":
        return clip_proba(proba, eps=config.clip_eps)
    if config.method == "shrink":
        return shrink_proba(proba, alpha=config.alpha, eps=config.clip_eps)
    if config.method == "logit_intercept":
        return logit_shift_proba(
            proba,
            intercept=config.logit_intercept,
            slope=1.0,
            eps=config.clip_eps,
        )
    if config.method == "platt":
        return logit_shift_proba(
            proba,
            intercept=config.platt_intercept,
            slope=config.platt_slope,
            eps=config.clip_eps,
        )
    raise ValueError(f"Unsupported postprocess method: {config.method}")


def add_prediction_columns(
    df: pd.DataFrame,
    proba_col: str,
    result_col: str | None = None,
    prefix: str = "",
) -> pd.DataFrame:
    out = df.copy()
    p = pd.to_numeric(out[proba_col], errors="coerce")
    out[f"{prefix}prediction_side"] = np.where(p >= 0.5, "HOME", "AWAY")
    out[f"{prefix}confidence"] = np.maximum(p, 1.0 - p)

    if result_col and result_col in out.columns:
        y = pd.to_numeric(out[result_col], errors="coerce")
        valid = p.notna() & y.isin([0, 1])
        out[f"{prefix}correct"] = pd.NA
        out.loc[valid, f"{prefix}correct"] = ((p.loc[valid] >= 0.5).astype(int) == y.loc[valid].astype(int)).astype(int)
        out[f"{prefix}brier"] = pd.NA
        out.loc[valid, f"{prefix}brier"] = (y.loc[valid] - p.loc[valid]) ** 2

    return out


def apply_postprocess_to_df(
    df: pd.DataFrame,
    config: PostprocessConfig,
    proba_col: str = PROBA_COL,
    result_col: str | None = RESULT_COL,
    overwrite: bool = False,
) -> pd.DataFrame:
    if proba_col not in df.columns:
        raise KeyError(f"Missing probability column: {proba_col}")

    out = df.copy()
    raw_col = f"{proba_col}_raw"
    post_col = proba_col if overwrite else f"{proba_col}_post"

    if raw_col not in out.columns:
        out[raw_col] = pd.to_numeric(out[proba_col], errors="coerce")

    out[post_col] = apply_proba_postprocess(out[raw_col], config)
    out["postprocess_name"] = config.label

    if not overwrite:
        out = add_prediction_columns(out, post_col, result_col=result_col, prefix="post_")
    else:
        out = add_prediction_columns(out, proba_col, result_col=result_col, prefix="")
        out["home_win_percent"] = (pd.to_numeric(out[proba_col], errors="coerce") * 100).round(2)
        out["away_win_percent"] = (100 - out["home_win_percent"]).round(2)

    return out


def load_finished_log(
    file_path: str | Path = DEFAULT_LOG_PATH,
    sheet_name: str = "Raw_Submission_Log",
    model_name: str | None = None,
    proba_col: str = PROBA_COL,
    result_col: str = RESULT_COL,
) -> pd.DataFrame:
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    if proba_col not in df.columns:
        raise KeyError(f"Missing probability column: {proba_col}")
    if result_col not in df.columns:
        raise KeyError(f"Missing result column: {result_col}")

    out = df.copy()
    out[proba_col] = pd.to_numeric(out[proba_col], errors="coerce")
    out[result_col] = pd.to_numeric(out[result_col], errors="coerce")
    out = out[out[proba_col].notna() & out[result_col].isin([0, 1])].copy()
    if model_name is not None:
        out = out[out[MODEL_COL].astype(str) == str(model_name)].copy()
    return out.reset_index(drop=True)


def metric_summary(
    df: pd.DataFrame,
    proba_col: str = PROBA_COL,
    result_col: str = RESULT_COL,
) -> dict[str, float | int]:
    p = pd.to_numeric(df[proba_col], errors="coerce")
    y = pd.to_numeric(df[result_col], errors="coerce")
    valid = p.notna() & y.isin([0, 1])
    p = p.loc[valid].astype(float)
    y = y.loc[valid].astype(int)
    if len(p) == 0:
        raise ValueError("No finished rows are available for metric calculation.")

    pred_home = (p >= 0.5).astype(int)
    brier_values = (y - p) ** 2
    return {
        "n": int(len(p)),
        "home_wins": int(y.sum()),
        "actual_home_rate": float(y.mean()),
        "avg_pred_home_proba": float(p.mean()),
        "bias_actual_minus_pred": float((y - p).mean()),
        "accuracy_05": float((pred_home == y).mean()),
        "brier": float(brier_values.mean()),
        "home_pred_count": int(pred_home.sum()),
        "away_pred_count": int(len(pred_home) - pred_home.sum()),
        "avg_confidence": float(np.maximum(p, 1.0 - p).mean()),
    }


def compare_raw_vs_post(
    df: pd.DataFrame,
    config: PostprocessConfig,
    proba_col: str = PROBA_COL,
    result_col: str = RESULT_COL,
) -> pd.DataFrame:
    post = apply_postprocess_to_df(
        df,
        config=config,
        proba_col=proba_col,
        result_col=result_col,
        overwrite=False,
    )
    raw_summary = metric_summary(post, proba_col=proba_col, result_col=result_col)
    post_summary = metric_summary(post, proba_col=f"{proba_col}_post", result_col=result_col)
    return pd.DataFrame(
        [
            {"case": "raw", **raw_summary},
            {"case": config.label, **post_summary},
        ]
    )


def evaluate_shrink_grid(
    df: pd.DataFrame,
    alphas=None,
    proba_col: str = PROBA_COL,
    result_col: str = RESULT_COL,
) -> pd.DataFrame:
    if alphas is None:
        alphas = np.linspace(0.0, 1.0, 101)

    rows = []
    for alpha in alphas:
        config = PostprocessConfig(method="shrink", alpha=float(alpha))
        summary = metric_summary(
            apply_postprocess_to_df(df, config, proba_col=proba_col, result_col=result_col, overwrite=False),
            proba_col=f"{proba_col}_post",
            result_col=result_col,
        )
        rows.append({"alpha": float(alpha), **summary})
    return pd.DataFrame(rows).sort_values(["brier", "accuracy_05"], ascending=[True, False]).reset_index(drop=True)


def fit_platt_config(
    df: pd.DataFrame,
    proba_col: str = PROBA_COL,
    result_col: str = RESULT_COL,
    C: float = 1.0,
) -> PostprocessConfig:
    p = clip_proba(df[proba_col])
    y = pd.to_numeric(df[result_col], errors="coerce").astype(int).to_numpy()
    model = LogisticRegression(C=C, solver="lbfgs")
    model.fit(logit(p).reshape(-1, 1), y)
    return PostprocessConfig(
        method="platt",
        platt_slope=float(model.coef_[0][0]),
        platt_intercept=float(model.intercept_[0]),
        name="platt_from_log",
    )


def calibration_by_probability_bin(
    df: pd.DataFrame,
    proba_col: str = PROBA_COL,
    result_col: str = RESULT_COL,
    bins=None,
) -> pd.DataFrame:
    if bins is None:
        bins = [0.0, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 1.0]
    work = df.copy()
    work[proba_col] = pd.to_numeric(work[proba_col], errors="coerce")
    work[result_col] = pd.to_numeric(work[result_col], errors="coerce")
    work = work[work[proba_col].notna() & work[result_col].isin([0, 1])].copy()
    work["proba_bin"] = pd.cut(work[proba_col], bins=bins, include_lowest=True)
    work["correct_05"] = ((work[proba_col] >= 0.5).astype(int) == work[result_col].astype(int)).astype(int)
    work["brier"] = (work[result_col] - work[proba_col]) ** 2
    return (
        work.groupby("proba_bin", observed=True)
        .agg(
            n=("game_id", "size"),
            avg_pred_home_proba=(proba_col, "mean"),
            actual_home_rate=(result_col, "mean"),
            accuracy_05=("correct_05", "mean"),
            brier=("brier", "mean"),
        )
        .assign(bias_actual_minus_pred=lambda x: x["actual_home_rate"] - x["avg_pred_home_proba"])
        .reset_index()
    )


def side_bias_summary(
    df: pd.DataFrame,
    proba_col: str = PROBA_COL,
    result_col: str = RESULT_COL,
) -> pd.DataFrame:
    work = df.copy()
    work[proba_col] = pd.to_numeric(work[proba_col], errors="coerce")
    work[result_col] = pd.to_numeric(work[result_col], errors="coerce")
    work = work[work[proba_col].notna() & work[result_col].isin([0, 1])].copy()
    work["prediction_side"] = np.where(work[proba_col] >= 0.5, "HOME", "AWAY")
    rows = []
    for side, group in work.groupby("prediction_side"):
        summary = metric_summary(group, proba_col=proba_col, result_col=result_col)
        rows.append({"prediction_side": side, **summary})
    return pd.DataFrame(rows).sort_values("prediction_side").reset_index(drop=True)


def exact_binomial_p_value(k: int, n: int, p0: float = 0.5, alternative: Literal["two-sided", "greater", "less"] = "two-sided") -> float:
    if not 0 <= k <= n:
        raise ValueError("k must satisfy 0 <= k <= n.")
    if not 0 < p0 < 1:
        raise ValueError("p0 must be between 0 and 1.")

    probs = [comb(n, i) * (p0**i) * ((1.0 - p0) ** (n - i)) for i in range(n + 1)]
    if alternative == "greater":
        return float(sum(probs[k:]))
    if alternative == "less":
        return float(sum(probs[: k + 1]))
    if alternative == "two-sided":
        observed = probs[k]
        return float(min(1.0, sum(prob for prob in probs if prob <= observed + 1e-15)))
    raise ValueError(f"Unsupported alternative: {alternative}")


def normal_mean_p_value(values, alternative: Literal["two-sided", "greater", "less"] = "two-sided") -> float:
    x = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float)
    if len(x) < 2:
        return float("nan")
    mean = float(x.mean())
    sd = float(x.std(ddof=1))
    if sd == 0:
        return 0.0 if mean != 0 else 1.0
    z = mean / (sd / sqrt(len(x)))
    upper = 0.5 * erfc(z / sqrt(2.0))
    lower = 1.0 - upper
    if alternative == "greater":
        return float(upper)
    if alternative == "less":
        return float(lower)
    if alternative == "two-sided":
        return float(min(1.0, 2.0 * min(upper, lower)))
    raise ValueError(f"Unsupported alternative: {alternative}")


def home_rate_test(
    df: pd.DataFrame,
    result_col: str = RESULT_COL,
    p0: float = 0.5,
) -> dict[str, float | int]:
    y = pd.to_numeric(df[result_col], errors="coerce")
    y = y[y.isin([0, 1])].astype(int)
    n = int(len(y))
    k = int(y.sum())
    if n == 0:
        raise ValueError("No finished result rows are available.")
    return {
        "n": n,
        "home_wins": k,
        "home_rate": float(k / n),
        "p_value_two_sided": exact_binomial_p_value(k, n, p0=p0, alternative="two-sided"),
        "p_value_home_rate_greater_than_p0": exact_binomial_p_value(k, n, p0=p0, alternative="greater"),
    }


def bias_test(
    df: pd.DataFrame,
    proba_col: str = PROBA_COL,
    result_col: str = RESULT_COL,
) -> dict[str, float | int]:
    p = pd.to_numeric(df[proba_col], errors="coerce")
    y = pd.to_numeric(df[result_col], errors="coerce")
    valid = p.notna() & y.isin([0, 1])
    residual = y.loc[valid].astype(float) - p.loc[valid].astype(float)
    return {
        "n": int(len(residual)),
        "bias_actual_minus_pred": float(residual.mean()),
        "p_value_two_sided_normal_approx": normal_mean_p_value(residual, alternative="two-sided"),
        "p_value_bias_greater_than_zero_normal_approx": normal_mean_p_value(residual, alternative="greater"),
    }


def build_postprocess_report(
    file_path: str | Path = DEFAULT_LOG_PATH,
    model_name: str | None = "elo_sp",
    shrink_alpha: float = 0.8,
) -> dict[str, pd.DataFrame | dict[str, float | int]]:
    log_df = load_finished_log(file_path=file_path, model_name=model_name)
    config = PostprocessConfig(method="shrink", alpha=shrink_alpha)
    return {
        "raw_summary": pd.DataFrame([metric_summary(log_df)]),
        "home_rate_test": home_rate_test(log_df),
        "bias_test": bias_test(log_df),
        "side_bias": side_bias_summary(log_df),
        "calibration_bins": calibration_by_probability_bin(log_df),
        "shrink_grid": evaluate_shrink_grid(log_df),
        "raw_vs_selected_postprocess": compare_raw_vs_post(log_df, config),
        "selected_config": asdict(config) | {"label": config.label},
    }


if __name__ == "__main__":
    report = build_postprocess_report()
    for name, value in report.items():
        print(f"\n=== {name} ===")
        if isinstance(value, pd.DataFrame):
            print(value.head(20).to_string(index=False))
        else:
            print(value)
