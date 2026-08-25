from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import contextlib
import io
import importlib

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import elo
import save_log as sv
import submit as sb
import utils as ut


ModelName = Literal["elo_full", "elo_sp", "elo_sp_l2"]

ROOT = Path(__file__).resolve().parent
TEAM_CODE_TO_NAME = sv.TEAM_CODE_TO_NAME


@dataclass(frozen=True)
class PredictionArtifacts:
    pred_df: pd.DataFrame
    pred_df_core: pd.DataFrame | None = None
    pred_df_elo: pd.DataFrame | None = None
    result_df: pd.DataFrame | None = None
    submit_df: pd.DataFrame | None = None
    model_name: str | None = None


def reload_project_modules() -> None:
    importlib.reload(ut)
    importlib.reload(elo)
    importlib.reload(sv)
    importlib.reload(sb)


class LoggedPredictionError(RuntimeError):
    def __init__(self, message: str, logs: str):
        super().__init__(message)
        self.logs = logs


def _mask_secret(value: str, head: int = 6, tail: int = 4) -> str:
    if not value:
        return ""
    if len(value) <= head + tail:
        return "*" * len(value)
    return f"{value[:head]}...{value[-tail:]}"


def get_api_debug_info() -> dict[str, str | int]:
    return {
        "utils_file": str(Path(ut.__file__).resolve()),
        "submit_file": str(Path(sb.__file__).resolve()),
        "utils_base_url": getattr(ut, "BASE_URL", ""),
        "submit_base_url": getattr(sb, "BASE_URL", ""),
        "utils_api_key": _mask_secret(getattr(ut, "API_KEY", "")),
        "submit_api_key": _mask_secret(getattr(sb, "API_KEY", "")),
        "utils_api_key_len": len(getattr(ut, "API_KEY", "")),
        "submit_api_key_len": len(getattr(sb, "API_KEY", "")),
        "utils_secret": _mask_secret(getattr(ut, "SECRET", "")),
        "submit_secret": _mask_secret(getattr(sb, "SECRET", "")),
        "utils_secret_len": len(getattr(ut, "SECRET", "")),
        "submit_secret_len": len(getattr(sb, "SECRET", "")),
    }


def load_prediction_data(date_str: str, sleep_sec: float = 0.12, verbose: bool = True) -> pd.DataFrame:
    pred_result = ut.build_prediction_data_for_date(
        date_str=date_str,
        sleep_sec=sleep_sec,
        verbose=verbose,
    )
    return pred_result["pred_df"]


def load_prediction_data_with_logs(date_str: str, sleep_sec: float = 0.12, verbose: bool = True) -> tuple[pd.DataFrame, str]:
    reload_project_modules()
    log_buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(log_buffer), contextlib.redirect_stderr(log_buffer):
            pred_df = load_prediction_data(date_str=date_str, sleep_sec=sleep_sec, verbose=verbose)
        return pred_df, log_buffer.getvalue()
    except Exception as exc:
        logs = log_buffer.getvalue()
        wrapped = LoggedPredictionError(str(exc), logs)
        wrapped.__cause__ = exc
        raise wrapped


def missing_value_count(df: pd.DataFrame) -> int:
    return int(df.isna().sum().sum())


def fill_starting_pitcher_features(pred_df: pd.DataFrame, target_year: int) -> pd.DataFrame:
    season_pit_df = pd.read_csv(ROOT / "seaon_pit_df.csv")
    return ut.fill_and_recompute_sp_features(
        pred_df=pred_df,
        season_pit_df=season_pit_df,
        target_year=target_year,
    )


def add_recent_and_elo_features(
    pred_df: pd.DataFrame,
    update_elo_snapshot: bool = True,
    recent_sleep_sec: float = 0.15,
    elo_sleep_sec: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None]:
    pred_df_core = ut.add_recent_flow_features_for_prediction(pred_df, sleep_sec=recent_sleep_sec)

    games_with_elo = None
    updated_snapshot = None
    if update_elo_snapshot:
        games_with_elo, updated_snapshot = elo.update_team_elo_snapshot_2026(
            sleep_sec=elo_sleep_sec,
            show=False,
        )

    pred_df_elo = elo.merge_elo(pred_df_core)
    return pred_df_core, pred_df_elo, games_with_elo, updated_snapshot


def _prepare_predict_frame(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        raise KeyError(f"모델 입력 컬럼이 없습니다: {missing_cols}")
    return df[feature_cols].copy().replace([np.inf, -np.inf], np.nan).fillna(0)


def _build_result_df(pred_df_elo: pd.DataFrame, proba: np.ndarray, model_name: str) -> pd.DataFrame:
    result_df = pred_df_elo[
        [
            "game_id",
            "game_date",
            "home_team_code",
            "away_team_code",
        ]
    ].copy()
    result_df["home_win_proba"] = proba
    result_df["model_name"] = model_name
    result_df["home_team"] = pd.to_numeric(result_df["home_team_code"], errors="coerce").map(TEAM_CODE_TO_NAME)
    result_df["away_team"] = pd.to_numeric(result_df["away_team_code"], errors="coerce").map(TEAM_CODE_TO_NAME)
    result_df["home_win_percent"] = (result_df["home_win_proba"] * 100).round(2)
    result_df["away_win_percent"] = (100 - result_df["home_win_percent"]).round(2)
    return result_df


def predict_with_saved_logit(pred_df_elo: pd.DataFrame, model_name: ModelName) -> pd.DataFrame:
    if model_name == "elo_full":
        model = joblib.load(ROOT / "logit_model_elo_full.pkl")
        feature_cols = joblib.load(ROOT / "feature_cols_elo_full.pkl")
    elif model_name == "elo_sp":
        model = joblib.load(ROOT / "logit_model_elo_sp.pkl")
        feature_cols = joblib.load(ROOT / "feature_cols_elo_sp.pkl")
    else:
        raise ValueError(f"Unsupported saved model: {model_name}")

    X_pred = _prepare_predict_frame(pred_df_elo, feature_cols)
    proba = model.predict_proba(X_pred)[:, 1]
    return _build_result_df(pred_df_elo, proba, model_name)


def predict_with_selected_logistic_l2(pred_df_elo: pd.DataFrame) -> pd.DataFrame:
    feature_cols = joblib.load(ROOT / "feature_cols_elo_sp.pkl")
    train_model_df = build_selected_training_frame()

    missing_train_cols = [c for c in feature_cols if c not in train_model_df.columns]
    if missing_train_cols:
        raise KeyError(f"elo_sp_l2 학습 컬럼이 없습니다: {missing_train_cols}")

    train_ready = train_model_df.dropna(subset=feature_cols + ["y_home_win"]).copy()
    X_train = train_ready[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    y_train = train_ready["y_home_win"].astype(int)

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=0.1,
                    penalty="l2",
                    solver="lbfgs",
                    max_iter=2000,
                    random_state=427,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)

    X_pred = _prepare_predict_frame(pred_df_elo, feature_cols)
    proba = model.predict_proba(X_pred)[:, 1]
    return _build_result_df(pred_df_elo, proba, "elo_sp_l2")


def run_prediction(pred_df_elo: pd.DataFrame, model_name: ModelName) -> pd.DataFrame:
    if model_name in {"elo_full", "elo_sp"}:
        return predict_with_saved_logit(pred_df_elo, model_name)
    if model_name == "elo_sp_l2":
        return predict_with_selected_logistic_l2(pred_df_elo)
    raise ValueError(f"지원하지 않는 모델입니다: {model_name}")


def make_submit_df(result_df: pd.DataFrame) -> pd.DataFrame:
    submit_df = result_df[["game_id", "home_win_proba"]].copy()
    submit_df["home_win_proba"] = submit_df["home_win_proba"].round(4) * 100
    return submit_df


def save_prediction_log(result_df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    return sv.append_prediction_log(
        pred_df=result_df,
        file_path=str(ROOT / "prediction_log.xlsx"),
        sheet_name="Raw_Submission_Log",
        dedup=True,
        model_name=model_name,
    )


def submit_predictions(submit_df: pd.DataFrame, sleep_sec: float = 0.2, verbose: bool = True) -> pd.DataFrame:
    return sb.submit_predictions_df(
        submit_df,
        path="prediction/savePrediction",
        sleep_sec=sleep_sec,
        verbose=verbose,
    )


def _make_team_game_log_selected(game_df: pd.DataFrame) -> pd.DataFrame:
    base_cols = ["game_id", "game_date", "season"]

    home = game_df[base_cols + ["home_team_code", "away_team_code", "home_score", "away_score", "elo_result_home"]].copy()
    home = home.rename(
        columns={
            "home_team_code": "team_code",
            "away_team_code": "opponent_team_code",
            "home_score": "runs_for",
            "away_score": "runs_against",
            "elo_result_home": "result",
        }
    )
    home["is_home"] = 1

    away = game_df[base_cols + ["away_team_code", "home_team_code", "away_score", "home_score", "elo_result_home"]].copy()
    away = away.rename(
        columns={
            "away_team_code": "team_code",
            "home_team_code": "opponent_team_code",
            "away_score": "runs_for",
            "home_score": "runs_against",
        }
    )
    away["result"] = 1.0 - pd.to_numeric(away["elo_result_home"], errors="coerce")
    away = away.drop(columns=["elo_result_home"])
    away["is_home"] = 0

    team_games = pd.concat([home, away], ignore_index=True)
    team_games["game_date"] = pd.to_datetime(team_games["game_date"], errors="coerce")
    team_games["team_code"] = team_games["team_code"].astype(str)
    team_games["opponent_team_code"] = team_games["opponent_team_code"].astype(str)
    team_games["runs_for"] = pd.to_numeric(team_games["runs_for"], errors="coerce")
    team_games["runs_against"] = pd.to_numeric(team_games["runs_against"], errors="coerce")
    team_games["result"] = pd.to_numeric(team_games["result"], errors="coerce")
    team_games["run_diff"] = team_games["runs_for"] - team_games["runs_against"]
    return team_games.sort_values(["team_code", "game_date", "game_id"]).reset_index(drop=True)


def _build_recent_features_selected(game_df: pd.DataFrame, windows=(5, 10)) -> pd.DataFrame:
    team_recent = _make_team_game_log_selected(game_df)
    grouped = team_recent.groupby(["team_code", "season"], group_keys=False)

    team_recent["prev_game_date"] = grouped["game_date"].shift(1)
    team_recent["rest_days"] = (team_recent["game_date"].dt.normalize() - team_recent["prev_game_date"].dt.normalize()).dt.days - 1
    team_recent["rest_days"] = team_recent["rest_days"].clip(lower=0)
    team_recent["prev_game_result"] = grouped["result"].shift(1)

    rolling_targets = {
        "result": "result_avg",
        "runs_for": "runs_for_avg",
        "runs_against": "runs_against_avg",
        "run_diff": "run_diff_avg",
    }
    for window in windows:
        for src_col, out_name in rolling_targets.items():
            out_col = f"recent_{window}_{out_name}"
            team_recent[out_col] = grouped[src_col].transform(
                lambda s, w=window: s.shift(1).rolling(w, min_periods=1).mean()
            )

    recent_feature_cols = ["game_id", "team_code", "rest_days", "prev_game_result"]
    recent_feature_cols += [c for c in team_recent.columns if c.startswith("recent_")]
    recent_features = team_recent[recent_feature_cols].copy()

    home_recent = recent_features.rename(
        columns={c: f"home_{c}" for c in recent_feature_cols if c not in ["game_id", "team_code"]}
    ).rename(columns={"team_code": "home_team_code"})
    away_recent = recent_features.rename(
        columns={c: f"away_{c}" for c in recent_feature_cols if c not in ["game_id", "team_code"]}
    ).rename(columns={"team_code": "away_team_code"})

    out = game_df.copy()
    out["home_team_code"] = out["home_team_code"].astype(str)
    out["away_team_code"] = out["away_team_code"].astype(str)
    out = out.merge(home_recent, on=["game_id", "home_team_code"], how="left")
    out = out.merge(away_recent, on=["game_id", "away_team_code"], how="left")

    diff_pairs = ["rest_days", "prev_game_result"] + [c for c in team_recent.columns if c.startswith("recent_")]
    for col in diff_pairs:
        out[f"{col}_diff"] = out[f"home_{col}"] - out[f"away_{col}"]
    return out


def _expected_home_score_selected(home_elo, away_elo, home_field_adv=25.0):
    return 1.0 / (1.0 + 10 ** (((away_elo) - (home_elo + home_field_adv)) / 400.0))


def _regress_to_mean_selected(old_elo, alpha=0.65, base_elo=1500.0):
    return base_elo + alpha * (old_elo - base_elo)


def _build_elo_features_selected(
    game_df: pd.DataFrame,
    alpha: float = 0.65,
    home_field_adv: float = 25.0,
    base_elo: float = 1500.0,
    k_factor: float = 20.0,
) -> pd.DataFrame:
    elo_df = game_df.copy().sort_values(["game_date", "game_id"]).reset_index(drop=True)
    elo_df["home_team_code"] = elo_df["home_team_code"].astype(str)
    elo_df["away_team_code"] = elo_df["away_team_code"].astype(str)

    ratings = {}
    last_seen_season = {}
    home_elo_pre_list = []
    away_elo_pre_list = []
    elo_exp_home_list = []
    home_elo_post_list = []
    away_elo_post_list = []

    for row in elo_df.itertuples(index=False):
        season = row.season
        home_team = row.home_team_code
        away_team = row.away_team_code

        for team in [home_team, away_team]:
            if team not in ratings:
                ratings[team] = base_elo
                last_seen_season[team] = season
            elif last_seen_season[team] != season:
                ratings[team] = _regress_to_mean_selected(ratings[team], alpha=alpha, base_elo=base_elo)
                last_seen_season[team] = season

        result_home = row.elo_result_home
        home_elo_pre = ratings[home_team]
        away_elo_pre = ratings[away_team]
        elo_exp_home = _expected_home_score_selected(home_elo_pre, away_elo_pre, home_field_adv=home_field_adv)

        elo_delta = k_factor * (result_home - elo_exp_home)
        home_elo_post = home_elo_pre + elo_delta
        away_elo_post = away_elo_pre - elo_delta

        ratings[home_team] = home_elo_post
        ratings[away_team] = away_elo_post
        last_seen_season[home_team] = season
        last_seen_season[away_team] = season

        home_elo_pre_list.append(home_elo_pre)
        away_elo_pre_list.append(away_elo_pre)
        elo_exp_home_list.append(elo_exp_home)
        home_elo_post_list.append(home_elo_post)
        away_elo_post_list.append(away_elo_post)

    elo_df["home_elo_pre"] = home_elo_pre_list
    elo_df["away_elo_pre"] = away_elo_pre_list
    elo_df["elo_diff"] = elo_df["home_elo_pre"] - elo_df["away_elo_pre"]
    elo_df["elo_exp_home"] = elo_exp_home_list
    elo_df["home_elo_post"] = home_elo_post_list
    elo_df["away_elo_post"] = away_elo_post_list
    return elo_df


def build_selected_training_frame() -> pd.DataFrame:
    train_base_df = pd.concat(
        [
            pd.read_csv(ROOT / "model_df_2023.csv"),
            pd.read_csv(ROOT / "model_df_2024.csv"),
            pd.read_csv(ROOT / "model_df_2025.csv"),
        ],
        ignore_index=True,
    )
    train_base_df["game_date"] = pd.to_datetime(train_base_df["game_date"], errors="coerce")
    for col in ["home_score", "away_score", "home_win", "y_home_win", "season"]:
        if col in train_base_df.columns:
            train_base_df[col] = pd.to_numeric(train_base_df[col], errors="coerce")

    train_base_df["elo_result_home"] = np.where(
        train_base_df["home_score"] > train_base_df["away_score"],
        1.0,
        np.where(train_base_df["home_score"] < train_base_df["away_score"], 0.0, 0.5),
    )
    train_base_df = train_base_df.sort_values(["game_date", "game_id"]).reset_index(drop=True)

    train_model_df = _build_recent_features_selected(train_base_df, windows=(5, 10))
    train_model_df = _build_elo_features_selected(train_model_df)

    selected_recent_cols = [
        "recent_5_result_avg_diff",
        "recent_5_runs_for_avg_diff",
        "recent_5_runs_against_avg_diff",
        "recent_5_run_diff_avg_diff",
        "recent_10_result_avg_diff",
        "recent_10_runs_for_avg_diff",
        "recent_10_runs_against_avg_diff",
        "recent_10_run_diff_avg_diff",
        "rest_days_diff",
        "prev_game_result_diff",
    ]
    selected_recent_cols = [c for c in selected_recent_cols if c in train_model_df.columns]
    train_model_df[selected_recent_cols] = train_model_df[selected_recent_cols].fillna(0)
    return train_model_df
