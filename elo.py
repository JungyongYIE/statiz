from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import importlib
import importlib.util
import sys
import types

import numpy as np
import pandas as pd
from IPython.display import display

try:
    from IPython.display import display as _ipython_display
except ImportError:
    ipython_module = types.ModuleType("IPython")
    display_module = types.ModuleType("IPython.display")
    display_module.display = print
    ipython_module.display = display_module
    sys.modules.setdefault("IPython", ipython_module)
    sys.modules.setdefault("IPython.display", display_module)

if importlib.util.find_spec("joblib") is None:
    joblib_module = types.ModuleType("joblib")
    joblib_module.load = lambda *args, **kwargs: []
    sys.modules.setdefault("joblib", joblib_module)

import utils as ut


UPDATE_SEASON_START = pd.Timestamp("2026-03-22")
DEFAULT_SNAPSHOT_FILE = "team_elo_snapshot_2026.csv"
BASE_ELO = 1500.0
K_FACTOR = 20.0
FINAL_HOME_FIELD_ADV = 25.0


def resolve_snapshot_path(snapshot_path: str | Path = DEFAULT_SNAPSHOT_FILE) -> Path:
    path = Path(snapshot_path)
    if path.is_absolute():
        return path
    return Path(__file__).resolve().parent / path


def get_update_target_date(tz: str = "Asia/Seoul") -> pd.Timestamp:
    """Return today's normalized local date as a timezone-naive Timestamp."""
    return pd.Timestamp.now(tz=tz).normalize().tz_localize(None)


def expected_home_score(home_elo: float, away_elo: float, home_field_adv: float = 0.0) -> float:
    return 1.0 / (1.0 + 10 ** ((away_elo - (home_elo + home_field_adv)) / 400.0))


def load_team_elo_snapshot(snapshot_path: str | Path = DEFAULT_SNAPSHOT_FILE) -> pd.DataFrame:
    snapshot = pd.read_csv(snapshot_path)
    snapshot["team_code"] = snapshot["team_code"].astype(str)
    snapshot["elo"] = pd.to_numeric(snapshot["elo"], errors="coerce")
    snapshot["latest_update_date"] = pd.to_datetime(snapshot["latest_update_date"], errors="coerce")
    return snapshot


def get_fetch_window(
    snapshot_df: pd.DataFrame,
    target_date: pd.Timestamp | None = None,
    season_start: pd.Timestamp = UPDATE_SEASON_START,
) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp | None]:
    latest_update = snapshot_df["latest_update_date"].dropna().max()
    latest_update = latest_update if pd.notna(latest_update) else None
    fetch_start = latest_update + timedelta(days=1) if pd.notna(latest_update) else season_start
    fetch_end = (target_date or get_update_target_date()) - timedelta(days=1)
    return fetch_start, fetch_end, latest_update


def fetch_new_games(fetch_start: pd.Timestamp, fetch_end: pd.Timestamp, sleep_sec: float = 0.2) -> pd.DataFrame:
    date_list = pd.date_range(fetch_start, fetch_end, freq="D").strftime("%Y-%m-%d").tolist()
    new_games_raw = ut.collect_games(date_list, sleep_sec=sleep_sec)
    if new_games_raw.empty:
        return pd.DataFrame()
    return prepare_new_games(new_games_raw)


def prepare_new_games(games_raw: pd.DataFrame) -> pd.DataFrame:
    games = ut.prepare_games_for_modeling(games_raw).copy()
    if games.empty:
        return games
    games["game_date"] = pd.to_datetime(games["game_date"], errors="coerce")
    games["home_team_code"] = games["home_team_code"].astype(str)
    games["away_team_code"] = games["away_team_code"].astype(str)
    games = games.sort_values(["game_date", "game_id"]).reset_index(drop=True)
    games["elo_result_home"] = np.where(
        games["home_score"] > games["away_score"],
        1.0,
        np.where(games["home_score"] < games["away_score"], 0.0, 0.5),
    )
    return games


def update_elo_without_regression(
    snapshot_df: pd.DataFrame,
    games_df: pd.DataFrame,
    home_field_adv: float = FINAL_HOME_FIELD_ADV,
    base_elo: float = BASE_ELO,
    k_factor: float = K_FACTOR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ratings = {str(row.team_code): float(row.elo) for row in snapshot_df.itertuples(index=False)}

    update_games = games_df.copy().sort_values(["game_date", "game_id"]).reset_index(drop=True)
    home_elo_pre_list = []
    away_elo_pre_list = []
    elo_exp_home_list = []
    home_elo_post_list = []
    away_elo_post_list = []

    for row in update_games.itertuples(index=False):
        home_team = str(row.home_team_code)
        away_team = str(row.away_team_code)

        home_elo_pre = ratings.get(home_team, base_elo)
        away_elo_pre = ratings.get(away_team, base_elo)
        elo_exp_home = expected_home_score(home_elo_pre, away_elo_pre, home_field_adv=home_field_adv)

        elo_delta = k_factor * (row.elo_result_home - elo_exp_home)
        home_elo_post = home_elo_pre + elo_delta
        away_elo_post = away_elo_pre - elo_delta

        ratings[home_team] = home_elo_post
        ratings[away_team] = away_elo_post

        home_elo_pre_list.append(home_elo_pre)
        away_elo_pre_list.append(away_elo_pre)
        elo_exp_home_list.append(elo_exp_home)
        home_elo_post_list.append(home_elo_post)
        away_elo_post_list.append(away_elo_post)

    update_games["home_elo_pre"] = home_elo_pre_list
    update_games["away_elo_pre"] = away_elo_pre_list
    update_games["elo_diff"] = update_games["home_elo_pre"] - update_games["away_elo_pre"]
    update_games["elo_exp_home"] = elo_exp_home_list
    update_games["home_elo_post"] = home_elo_post_list
    update_games["away_elo_post"] = away_elo_post_list

    updated_snapshot = pd.DataFrame({"team_code": list(ratings.keys()), "elo": list(ratings.values())})
    updated_snapshot = updated_snapshot.sort_values("team_code").reset_index(drop=True)
    return update_games, updated_snapshot


def save_updated_snapshot(
    updated_snapshot: pd.DataFrame,
    games_with_elo: pd.DataFrame,
    snapshot_path: str | Path = DEFAULT_SNAPSHOT_FILE,
) -> tuple[pd.DataFrame, str | None]:
    latest_update = pd.to_datetime(games_with_elo["game_date"], errors="coerce").max()
    latest_update_str = pd.Timestamp(latest_update).strftime("%Y-%m-%d") if pd.notna(latest_update) else None

    output_df = updated_snapshot.copy()
    output_df["latest_update_date"] = latest_update_str
    output_df.to_csv(snapshot_path, index=False, encoding="utf-8-sig")
    return output_df, latest_update_str


def update_team_elo_snapshot_2026(
    snapshot_path: str | Path = DEFAULT_SNAPSHOT_FILE,
    target_date: pd.Timestamp | None = None,
    season_start: pd.Timestamp = UPDATE_SEASON_START,
    home_field_adv: float = FINAL_HOME_FIELD_ADV,
    base_elo: float = BASE_ELO,
    k_factor: float = K_FACTOR,
    sleep_sec: float = 0.2,
    reload_utils: bool = True,
    show: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Update team_elo_snapshot_2026.csv with games through yesterday."""
    if reload_utils:
        importlib.reload(ut)

    snapshot_path = resolve_snapshot_path(snapshot_path)
    snapshot = load_team_elo_snapshot(snapshot_path)
    target_date = target_date or get_update_target_date()
    fetch_start, fetch_end, latest_update = get_fetch_window(snapshot, target_date, season_start)

    print("UPDATE_TARGET_DATE:", target_date.strftime("%Y-%m-%d"))
    if fetch_start > fetch_end:
        print("No new games to update after", latest_update)
        return pd.DataFrame(), snapshot

    new_games = fetch_new_games(fetch_start, fetch_end, sleep_sec=sleep_sec)
    if new_games.empty:
        print(f"No completed games found from {fetch_start:%Y-%m-%d} to {fetch_end:%Y-%m-%d}")
        return pd.DataFrame(), snapshot

    games_with_elo, updated_snapshot = update_elo_without_regression(
        snapshot_df=snapshot[["team_code", "elo"]].copy(),
        games_df=new_games,
        home_field_adv=home_field_adv,
        base_elo=base_elo,
        k_factor=k_factor,
    )
    updated_snapshot, latest_update_new = save_updated_snapshot(updated_snapshot, games_with_elo, snapshot_path)

    if show:
        display_update_result(games_with_elo, updated_snapshot)

    print(f"saved: {snapshot_path}")
    print("updated latest_update_date:", latest_update_new)
    return games_with_elo, updated_snapshot


def display_update_result(games_with_elo: pd.DataFrame, snapshot_df: pd.DataFrame, tail_n: int = 20) -> None:
    game_cols = [
        "game_id",
        "game_date",
        "home_team_code",
        "away_team_code",
        "elo_result_home",
        "home_elo_pre",
        "away_elo_pre",
        "elo_diff",
        "home_elo_post",
        "away_elo_post",
    ]
    try:
        from IPython.display import display

        display(games_with_elo[game_cols].tail(tail_n))
        display(snapshot_df)
    except ImportError:
        print(games_with_elo[game_cols].tail(tail_n))
        print(snapshot_df)


if __name__ == "__main__":
    update_team_elo_snapshot_2026()


def merge_elo(pred_df_core):
    team_elo_snapshot = pd.read_csv("team_elo_snapshot_2026.csv")
    team_elo_snapshot["team_code"] = team_elo_snapshot["team_code"].astype(str)
    team_elo_snapshot["elo"] = pd.to_numeric(team_elo_snapshot["elo"], errors="coerce")

    home_elo_df = team_elo_snapshot.rename(columns={"team_code": "home_team_code", "elo": "home_elo_pre"})
    away_elo_df = team_elo_snapshot.rename(columns={"team_code": "away_team_code", "elo": "away_elo_pre"})

    pred_df_core["home_team_code"] = pred_df_core["home_team_code"].astype(str)
    pred_df_core["away_team_code"] = pred_df_core["away_team_code"].astype(str)
    pred_df_elo = pred_df_core.merge(home_elo_df, on="home_team_code", how="left")
    pred_df_elo = pred_df_elo.merge(away_elo_df, on="away_team_code", how="left")

    ELO_HOME_FIELD_ADV = 25
    pred_df_elo["elo_diff"] = pred_df_elo["home_elo_pre"] - pred_df_elo["away_elo_pre"]
    pred_df_elo["elo_exp_home"] = 1.0 / (
        1.0 + 10 ** (((pred_df_elo["away_elo_pre"]) - (pred_df_elo["home_elo_pre"] + ELO_HOME_FIELD_ADV)) / 400.0)
    )

    display(pred_df_elo[["game_id", "home_team_code", "away_team_code", "home_elo_pre", "away_elo_pre", "elo_diff", "elo_exp_home"]])
    print("missing elo rows:", int(pred_df_elo[["home_elo_pre", "away_elo_pre", "elo_diff", "elo_exp_home"]].isna().any(axis=1).sum()))
    return pred_df_elo
