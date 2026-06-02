from IPython.display import display
import time
import hmac
import hashlib
import urllib.request
from urllib.parse import quote
import json
import pandas as pd
import time
from typing import Any, Dict, Iterable, List, Optional
from pathlib import Path



from utils import (
    api_get,
    collect_games,
    collect_lineups,
    collect_rosters,
    convert_ip_baseball_to_decimal,
    fetch_player_season,
    to_df_from_list_or_numeric_dict,
)

API_KEY = "9cc2976dc30ff6190b7b0485536cc483"
SECRET  = "61f2cb7a045b539c7a530af70a5be50cbff571f69e6dd8fbaa3dcc1fb8835ac8"
BASE_URL = "https://api.statiz.co.kr/baseballApi"


# ?ш린??紐낆꽭?쒖뿉 留욎떠???섏젙.
METHOD = "GET"
PATH   = "prediction/playerSeason"
QUERY  = {"p_no": "14170"}

def normalize_query(params: dict) -> str:
    safe = "-_.!~*'()"
    return "&".join(
        f"{quote(str(k), safe=safe)}={quote(str(params[k]), safe=safe)}"
        for k in sorted(params.keys())
    )

def make_signature(secret: str, payload: str) -> str:
    return hmac.new(
        secret.encode("utf-8"),
        payload.encode("utf-8"),
        hashlib.sha256
    ).hexdigest()

normalized = normalize_query(QUERY)
timestamp = str(int(time.time()))

payload = f"{METHOD}|{PATH}|{normalized}|{timestamp}"
signature = make_signature(SECRET, payload)

url = f"https://api.statiz.co.kr/baseballApi/{PATH}?{normalized}"

headers = {
    "X-API-KEY": API_KEY,
    "X-TIMESTAMP": timestamp,
    "X-SIGNATURE": signature,
}

req = urllib.request.Request(url, method=METHOD, headers=headers)

with urllib.request.urlopen(req, timeout=30) as resp:
    body = resp.read().decode("utf-8")


BULLPEN_DIFF_FEATURE_COLUMNS = [
    "recent_bullpen_fip_7d_diff",
    "recent_bullpen_whip_7d_diff",
    "bullpen_np_3d_diff",
    "bullpen_np_7d_diff",
    "bullpen_ip_7d_diff",
    "bullpen_b2b_count_diff",
    "bullpen_pitcher_count_7d_diff",
]

BULLPEN_QUALITY_AVAILABILITY_COLUMNS = [
    "home_recent_bullpen_quality_available_7d",
    "away_recent_bullpen_quality_available_7d",
    "recent_bullpen_quality_available_7d_diff",
]


def prepare_bullpen_appearance_df(
    appearance_df: pd.DataFrame,
    *,
    game_id_col: str = "game_id",
    game_date_col: str = "game_date",
    team_code_col: str = "team_code",
    player_id_col: str = "player_id",
    player_name_col: str = "player_name",
    ip_col: str = "IP",
    np_col: str = "NP",
    er_col: str = "ER",
    bb_col: str = "BB",
    hr_col: str = "HR",
    h_col: str = "H",
    k_col: str = "K",
    is_starting_pitcher_col: Optional[str] = "is_starting_pitcher",
    gs_col: Optional[str] = None,
) -> pd.DataFrame:
    if appearance_df.empty:
        return pd.DataFrame()

    df = appearance_df.copy()

    rename_map = {
        game_id_col: "game_id",
        game_date_col: "game_date",
        team_code_col: "team_code",
        player_id_col: "player_id",
        ip_col: "IP",
        np_col: "NP",
        er_col: "ER",
        bb_col: "BB",
        hr_col: "HR",
        h_col: "H",
        k_col: "K",
    }
    if player_name_col and player_name_col in df.columns:
        rename_map[player_name_col] = "player_name"
    if is_starting_pitcher_col and is_starting_pitcher_col in df.columns:
        rename_map[is_starting_pitcher_col] = "is_starting_pitcher"
    if gs_col and gs_col in df.columns:
        rename_map[gs_col] = "GS"

    df = df.rename(columns=rename_map).copy()

    required_cols = ["game_id", "game_date", "team_code", "player_id", "IP", "NP", "ER", "BB", "HR", "H", "K"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"appearance_df에 필요한 컬럼이 없습니다: {missing_cols}")

    df["game_id"] = df["game_id"].astype(str)
    df["team_code"] = df["team_code"].astype(str)
    df["player_id"] = df["player_id"].astype(str)
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")

    if "player_name" not in df.columns:
        df["player_name"] = pd.NA

    df["IP"] = df["IP"].astype(str)
    df["IP_decimal"] = df["IP"].apply(convert_ip_baseball_to_decimal)

    for col in ["IP_decimal", "NP", "ER", "BB", "HR", "H", "K"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    if "is_starting_pitcher" in df.columns:
        df["is_starting_pitcher"] = (
            df["is_starting_pitcher"]
            .replace({"Y": 1, "N": 0, True: 1, False: 0})
        )
        df["is_starting_pitcher"] = pd.to_numeric(df["is_starting_pitcher"], errors="coerce").fillna(0).astype(int)
    elif "GS" in df.columns:
        df["GS"] = pd.to_numeric(df["GS"], errors="coerce").fillna(0)
        df["is_starting_pitcher"] = (df["GS"] > 0).astype(int)
    else:
        raise ValueError("선발/불펜 구분용 컬럼이 필요합니다. is_starting_pitcher_col 또는 gs_col을 지정해주세요.")

    df["is_bullpen"] = (df["is_starting_pitcher"] == 0).astype(int)

    subset_cols = ["game_id", "team_code", "player_id"]
    df = df.drop_duplicates(subset=subset_cols).reset_index(drop=True)
    return df


def _build_bullpen_b2b_flags(bullpen_df: pd.DataFrame) -> pd.DataFrame:
    if bullpen_df.empty:
        return bullpen_df.copy()

    df = bullpen_df.copy()
    df = df[df["is_bullpen"] == 1].copy()
    if df.empty:
        return df

    df = df.sort_values(["team_code", "player_id", "game_date", "game_id"]).reset_index(drop=True)
    df["prev_game_date"] = df.groupby(["team_code", "player_id"])["game_date"].shift(1)
    df["days_since_prev"] = (
        df["game_date"].dt.normalize() - df["prev_game_date"].dt.normalize()
    ).dt.days
    df["is_b2b"] = (df["days_since_prev"] == 1).astype(int)
    return df


def build_recent_bullpen_team_features(
    games_df: pd.DataFrame,
    appearance_df: pd.DataFrame,
    *,
    fip_constant: float = 3.2,
) -> pd.DataFrame:
    if games_df.empty:
        return pd.DataFrame()

    bullpen_df = prepare_bullpen_appearance_df(appearance_df)
    if bullpen_df.empty or "is_bullpen" not in bullpen_df.columns:
        bullpen_df = pd.DataFrame(columns=[
            "game_id", "game_date", "team_code", "player_id",
            "IP", "IP_decimal", "NP", "ER", "BB", "HR", "H", "K",
            "is_starting_pitcher", "is_bullpen"
        ])
    bullpen_df = _build_bullpen_b2b_flags(bullpen_df)
    if bullpen_df.empty or "is_bullpen" not in bullpen_df.columns:
        bullpen_df = pd.DataFrame(columns=[
            "game_id", "game_date", "team_code", "player_id",
            "IP", "IP_decimal", "NP", "ER", "BB", "HR", "H", "K",
            "is_starting_pitcher", "is_bullpen", "is_b2b"
        ])
    else:
        bullpen_df = bullpen_df[bullpen_df["is_bullpen"] == 1].copy()

    games = games_df.copy()
    games["game_id"] = games["game_id"].astype(str)
    games["game_date"] = pd.to_datetime(games["game_date"], errors="coerce")
    games["home_team_code"] = games["home_team_code"].astype(str)
    games["away_team_code"] = games["away_team_code"].astype(str)

    records: List[Dict[str, Any]] = []

    for game in games.itertuples(index=False):
        game_id = str(getattr(game, "game_id"))
        game_date = pd.to_datetime(getattr(game, "game_date"))
        row: Dict[str, Any] = {
            "game_id": game_id,
            "game_date": game_date,
            "home_team_code": str(getattr(game, "home_team_code")),
            "away_team_code": str(getattr(game, "away_team_code")),
        }
        if hasattr(game, "season"):
            row["season"] = getattr(game, "season")

        for side, team_code in [("home", row["home_team_code"]), ("away", row["away_team_code"])]:
            team_hist = bullpen_df[
                (bullpen_df["team_code"] == team_code) &
                (bullpen_df["game_date"] < game_date)
            ].copy()

            hist_3d = team_hist[
                team_hist["game_date"] >= (game_date.normalize() - pd.Timedelta(days=3))
            ].copy()
            hist_7d = team_hist[
                team_hist["game_date"] >= (game_date.normalize() - pd.Timedelta(days=7))
            ].copy()

            row[f"{side}_bullpen_np_3d"] = float(hist_3d["NP"].sum()) if "NP" in hist_3d.columns else 0.0
            row[f"{side}_bullpen_np_7d"] = float(hist_7d["NP"].sum()) if "NP" in hist_7d.columns else 0.0
            row[f"{side}_bullpen_ip_7d"] = float(hist_7d["IP_decimal"].sum()) if "IP_decimal" in hist_7d.columns else 0.0
            row[f"{side}_bullpen_b2b_count"] = float(hist_7d["is_b2b"].sum()) if "is_b2b" in hist_7d.columns else 0.0
            row[f"{side}_bullpen_pitcher_count_7d"] = float(hist_7d["player_id"].nunique()) if "player_id" in hist_7d.columns else 0.0

            ip_7d = float(hist_7d["IP_decimal"].sum()) if "IP_decimal" in hist_7d.columns else 0.0
            bb_7d = float(hist_7d["BB"].sum()) if "BB" in hist_7d.columns else 0.0
            h_7d = float(hist_7d["H"].sum()) if "H" in hist_7d.columns else 0.0
            hr_7d = float(hist_7d["HR"].sum()) if "HR" in hist_7d.columns else 0.0
            k_7d = float(hist_7d["K"].sum()) if "K" in hist_7d.columns else 0.0

            if ip_7d > 0:
                row[f"{side}_recent_bullpen_quality_available_7d"] = 1
                row[f"{side}_recent_bullpen_whip_7d"] = (bb_7d + h_7d) / ip_7d
                row[f"{side}_recent_bullpen_fip_7d"] = ((13 * hr_7d) + (3 * bb_7d) - (2 * k_7d)) / ip_7d + fip_constant
            else:
                row[f"{side}_recent_bullpen_quality_available_7d"] = 0
                row[f"{side}_recent_bullpen_whip_7d"] = pd.NA
                row[f"{side}_recent_bullpen_fip_7d"] = pd.NA

        row["recent_bullpen_quality_available_7d_diff"] = (
            row["home_recent_bullpen_quality_available_7d"] -
            row["away_recent_bullpen_quality_available_7d"]
        )
        if (
            row["home_recent_bullpen_quality_available_7d"] == 1 and
            row["away_recent_bullpen_quality_available_7d"] == 1
        ):
            row["recent_bullpen_fip_7d_diff"] = row["home_recent_bullpen_fip_7d"] - row["away_recent_bullpen_fip_7d"]
            row["recent_bullpen_whip_7d_diff"] = row["home_recent_bullpen_whip_7d"] - row["away_recent_bullpen_whip_7d"]
        else:
            row["recent_bullpen_fip_7d_diff"] = pd.NA
            row["recent_bullpen_whip_7d_diff"] = pd.NA
        row["bullpen_np_3d_diff"] = row["home_bullpen_np_3d"] - row["away_bullpen_np_3d"]
        row["bullpen_np_7d_diff"] = row["home_bullpen_np_7d"] - row["away_bullpen_np_7d"]
        row["bullpen_ip_7d_diff"] = row["home_bullpen_ip_7d"] - row["away_bullpen_ip_7d"]
        row["bullpen_b2b_count_diff"] = row["home_bullpen_b2b_count"] - row["away_bullpen_b2b_count"]
        row["bullpen_pitcher_count_7d_diff"] = row["home_bullpen_pitcher_count_7d"] - row["away_bullpen_pitcher_count_7d"]

        records.append(row)

    out = pd.DataFrame(records)
    for col in BULLPEN_DIFF_FEATURE_COLUMNS:
        if col not in out.columns:
            out[col] = 0.0
    for col in BULLPEN_QUALITY_AVAILABILITY_COLUMNS:
        if col not in out.columns:
            out[col] = 0
    return out


def collect_recent_bullpen_feature_frame_by_dates(
    date_list: List[str],
    appearance_df: pd.DataFrame,
    *,
    sleep_sec: float = 0.2,
    fip_constant: float = 3.2,
) -> Dict[str, pd.DataFrame]:
    games_df = collect_games(date_list=date_list, sleep_sec=sleep_sec)
    bullpen_features = build_recent_bullpen_team_features(
        games_df=games_df,
        appearance_df=appearance_df,
        fip_constant=fip_constant,
    )
    return {
        "games": games_df,
        "bullpen_features": bullpen_features,
    }


def append_recent_bullpen_features(
    model_df: pd.DataFrame,
    appearance_df: pd.DataFrame,
    *,
    game_id_col: str = "game_id",
    game_date_col: str = "game_date",
    home_team_code_col: str = "home_team_code",
    away_team_code_col: str = "away_team_code",
    season_col: Optional[str] = "season",
    fip_constant: float = 3.2,
) -> pd.DataFrame:
    """
    model_df_2023.csv 같은 경기 단위 데이터프레임에 최근 불펜 피처를 붙여 반환한다.

    Parameters
    ----------
    model_df : pd.DataFrame
        경기 단위 데이터프레임. 최소한 game_id, game_date,
        home_team_code, away_team_code 컬럼이 있어야 한다.
    appearance_df : pd.DataFrame
        투수 등판 단위 데이터프레임.
    game_id_col, game_date_col, home_team_code_col, away_team_code_col : str
        model_df 쪽 컬럼명 매핑.
    season_col : str | None, default "season"
        season 컬럼이 있으면 함께 전달하고, 없으면 생략한다.
    fip_constant : float, default 3.2
        최근 불펜 FIP 계산 상수.

    Returns
    -------
    pd.DataFrame
        원본 model_df에 최근 불펜 피처가 left join 된 데이터프레임.
    """
    if model_df.empty:
        return model_df.copy()

    games = model_df.copy()
    rename_map = {
        game_id_col: "game_id",
        game_date_col: "game_date",
        home_team_code_col: "home_team_code",
        away_team_code_col: "away_team_code",
    }
    if season_col and season_col in games.columns:
        rename_map[season_col] = "season"

    games = games.rename(columns=rename_map).copy()

    required_cols = ["game_id", "game_date", "home_team_code", "away_team_code"]
    missing_cols = [c for c in required_cols if c not in games.columns]
    if missing_cols:
        raise ValueError(f"model_df에 필요한 컬럼이 없습니다: {missing_cols}")

    games["game_id"] = games["game_id"].astype(str)
    games["game_date"] = pd.to_datetime(games["game_date"], errors="coerce")
    games["home_team_code"] = games["home_team_code"].astype(str)
    games["away_team_code"] = games["away_team_code"].astype(str)

    bullpen_features = build_recent_bullpen_team_features(
        games_df=games[["game_id", "game_date", "home_team_code", "away_team_code"] + (["season"] if "season" in games.columns else [])],
        appearance_df=appearance_df,
        fip_constant=fip_constant,
    )

    feature_cols = [
        c for c in bullpen_features.columns
        if c not in {"game_date", "home_team_code", "away_team_code", "season"}
    ]
    bullpen_features = bullpen_features[feature_cols].copy()

    out = model_df.copy()
    out[game_id_col] = out[game_id_col].astype(str)
    out = out.merge(
        bullpen_features,
        left_on=game_id_col,
        right_on="game_id",
        how="left",
    )

    if game_id_col != "game_id":
        out = out.drop(columns=["game_id"])

    return out


def load_local_appearance_df(
    file_path: str,
    *,
    file_type: Optional[str] = None,
    game_id_col: str = "game_id",
    game_date_col: str = "game_date",
    team_code_col: str = "team_code",
    player_id_col: str = "player_id",
    player_name_col: str = "player_name",
    ip_col: str = "IP",
    np_col: str = "NP",
    er_col: str = "ER",
    bb_col: str = "BB",
    hr_col: str = "HR",
    h_col: str = "H",
    k_col: str = "K",
    is_starting_pitcher_col: Optional[str] = "is_starting_pitcher",
    gs_col: Optional[str] = None,
    sheet_name: int | str = 0,
) -> pd.DataFrame:
    """
    로컬 파일(csv/xlsx/parquet)에서 투수 등판 로그를 읽어
    prepare_bullpen_appearance_df 표준 포맷으로 반환한다.

    Notes
    -----
    이 함수는 파일을 "읽고 표준화"만 한다.
    실제 파일 안에는 경기별 투수 등판 로그가 있어야 하며,
    최소한 game_id, game_date, team_code, player_id, IP, NP,
    ER, BB, HR, H, K, 그리고 is_starting_pitcher 또는 GS가 필요하다.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"로컬 appearance 파일을 찾을 수 없습니다: {file_path}")

    inferred_type = (file_type or path.suffix.lstrip(".")).lower()

    if inferred_type == "csv":
        raw_df = pd.read_csv(path)
    elif inferred_type in {"xlsx", "xls"}:
        raw_df = pd.read_excel(path, sheet_name=sheet_name)
    elif inferred_type == "parquet":
        raw_df = pd.read_parquet(path)
    else:
        raise ValueError(
            "지원하지 않는 파일 형식입니다. csv, xlsx/xls, parquet 중 하나를 사용해주세요."
        )

    return prepare_bullpen_appearance_df(
        raw_df,
        game_id_col=game_id_col,
        game_date_col=game_date_col,
        team_code_col=team_code_col,
        player_id_col=player_id_col,
        player_name_col=player_name_col,
        ip_col=ip_col,
        np_col=np_col,
        er_col=er_col,
        bb_col=bb_col,
        hr_col=hr_col,
        h_col=h_col,
        k_col=k_col,
        is_starting_pitcher_col=is_starting_pitcher_col,
        gs_col=gs_col,
    )


def append_recent_bullpen_features_from_file(
    model_df: pd.DataFrame,
    appearance_file_path: str,
    *,
    appearance_file_type: Optional[str] = None,
    appearance_game_id_col: str = "game_id",
    appearance_game_date_col: str = "game_date",
    appearance_team_code_col: str = "team_code",
    appearance_player_id_col: str = "player_id",
    appearance_player_name_col: str = "player_name",
    appearance_ip_col: str = "IP",
    appearance_np_col: str = "NP",
    appearance_er_col: str = "ER",
    appearance_bb_col: str = "BB",
    appearance_hr_col: str = "HR",
    appearance_h_col: str = "H",
    appearance_k_col: str = "K",
    appearance_is_starting_pitcher_col: Optional[str] = "is_starting_pitcher",
    appearance_gs_col: Optional[str] = None,
    appearance_sheet_name: int | str = 0,
    model_game_id_col: str = "game_id",
    model_game_date_col: str = "game_date",
    model_home_team_code_col: str = "home_team_code",
    model_away_team_code_col: str = "away_team_code",
    model_season_col: Optional[str] = "season",
    fip_constant: float = 3.2,
) -> pd.DataFrame:
    """
    로컬 등판 로그 파일을 읽어 appearance_df를 만든 뒤,
    model_df에 최근 불펜 피처를 붙여 반환한다.
    """
    appearance_df = load_local_appearance_df(
        appearance_file_path,
        file_type=appearance_file_type,
        game_id_col=appearance_game_id_col,
        game_date_col=appearance_game_date_col,
        team_code_col=appearance_team_code_col,
        player_id_col=appearance_player_id_col,
        player_name_col=appearance_player_name_col,
        ip_col=appearance_ip_col,
        np_col=appearance_np_col,
        er_col=appearance_er_col,
        bb_col=appearance_bb_col,
        hr_col=appearance_hr_col,
        h_col=appearance_h_col,
        k_col=appearance_k_col,
        is_starting_pitcher_col=appearance_is_starting_pitcher_col,
        gs_col=appearance_gs_col,
        sheet_name=appearance_sheet_name,
    )

    return append_recent_bullpen_features(
        model_df,
        appearance_df,
        game_id_col=model_game_id_col,
        game_date_col=model_game_date_col,
        home_team_code_col=model_home_team_code_col,
        away_team_code_col=model_away_team_code_col,
        season_col=model_season_col,
        fip_constant=fip_constant,
    )


def fetch_player_day(
    player_id: str,
    *,
    year: Optional[int] = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    /prediction/playerDay 명세를 사용해 특정 선수의 날짜별 경기 기록을 불러온다.
    투수 기준으로는 경기별 등판 로그 원천으로 활용할 수 있다.
    """
    query: Dict[str, Any] = {"p_no": str(player_id)}
    if year is not None:
        query["year"] = int(year)

    data = api_get("prediction/playerDay", query, verbose=verbose)
    df = to_df_from_list_or_numeric_dict(data, list_key="list")

    if df.empty and isinstance(data, dict):
        candidate_keys = [k for k in data.keys() if str(k) not in {"result_cd", "result_msg", "update_time"}]
        for key in candidate_keys:
            if isinstance(data.get(key), list):
                df = pd.DataFrame(data[key])
                if not df.empty:
                    break
            if isinstance(data.get(key), dict):
                rows = [v for k2, v in data[key].items() if str(k2).isdigit() and isinstance(v, dict)]
                if rows:
                    df = pd.DataFrame(rows)
                    break

    if df.empty:
        return df

    rename_map = {
        "s_no": "game_id",
        "gameDate": "game_date",
        "p_no": "player_id",
        "t_code": "team_code",
        "SO": "K",
    }
    df = df.rename(columns=rename_map).copy()
    df["player_id"] = str(player_id)

    if "game_date" in df.columns:
        df["game_date"] = pd.to_datetime(df["game_date"], unit="s", errors="coerce")

    return df


def collect_pitcher_ids_from_rosters(
    roster_df: pd.DataFrame,
    *,
    sleep_sec: float = 0.2,
) -> pd.DataFrame:
    """
    로스터 선수 목록에서 playerSeason을 재조회해 투수만 추린다.
    반환 컬럼: player_id, player_name
    """
    if roster_df.empty:
        return pd.DataFrame(columns=["player_id", "player_name"])

    records: List[Dict[str, Any]] = []
    dedup = (
        roster_df[["player_id", "player_name"]]
        .dropna(subset=["player_id"])
        .copy()
    )
    dedup["player_id"] = dedup["player_id"].astype(str)
    dedup = dedup.drop_duplicates(subset=["player_id"]).reset_index(drop=True)

    for row in dedup.itertuples(index=False):
        player_id = str(getattr(row, "player_id"))
        player_name = getattr(row, "player_name", pd.NA)
        try:
            res = fetch_player_season(player_id)
            if not res["pitcher"].empty:
                records.append({
                    "player_id": player_id,
                    "player_name": player_name,
                })
            print(f"[pitcher-id] {player_id}: OK")
        except Exception as e:
            print(f"[pitcher-id] {player_id}: ERROR -> {e}")
        time.sleep(sleep_sec)

    if not records:
        return pd.DataFrame(columns=["player_id", "player_name"])

    return pd.DataFrame(records).drop_duplicates(subset=["player_id"]).reset_index(drop=True)


def collect_pitcher_appearance_df_by_dates(
    date_list: List[str],
    *,
    team_codes: Optional[List[str]] = None,
    roster_code: str = "1",
    lookback_days: int = 7,
    sleep_sec: float = 0.2,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    날짜 목록 기준으로 로스터를 수집하고, 로스터 선수 중 투수만 골라
    /prediction/playerDay 경기 로그를 합쳐 appearance_df를 만든다.
    """
    if not date_list:
        return pd.DataFrame()

    norm_dates = sorted({str(d) for d in date_list})
    if team_codes is None:
        games_df = collect_games(norm_dates, sleep_sec=sleep_sec)
        if games_df.empty:
            return pd.DataFrame()
        home_codes = games_df["home_team_code"].astype(str).tolist() if "home_team_code" in games_df.columns else []
        away_codes = games_df["away_team_code"].astype(str).tolist() if "away_team_code" in games_df.columns else []
        team_codes = sorted(set(home_codes + away_codes))

    roster_frames: List[pd.DataFrame] = []
    for dt in norm_dates:
        try:
            roster_df = collect_rosters(
                date=dt,
                team_codes=team_codes,
                code=roster_code,
                sleep_sec=sleep_sec,
            )
            if not roster_df.empty:
                roster_frames.append(roster_df)
            if verbose:
                print(f"[appearance-roster] {dt}: {len(roster_df)} rows")
        except Exception as e:
            print(f"[appearance-roster] {dt}: ERROR -> {e}")

    if roster_frames:
        roster_all = pd.concat(roster_frames, ignore_index=True).drop_duplicates()
    else:
        roster_all = pd.DataFrame(columns=["snapshot_date", "team_code", "player_id", "player_name", "join_date"])

    pitcher_ids_df = collect_pitcher_ids_from_rosters(roster_all, sleep_sec=sleep_sec)
    if pitcher_ids_df.empty:
        return pd.DataFrame()

    years = sorted({pd.to_datetime(dt).year for dt in norm_dates})
    min_date = pd.to_datetime(min(norm_dates), errors="coerce")
    max_date = pd.to_datetime(max(norm_dates), errors="coerce")
    min_keep_date = min_date - pd.Timedelta(days=max(lookback_days, 0))

    appearance_frames: List[pd.DataFrame] = []
    pitcher_name_map = pitcher_ids_df.set_index("player_id")["player_name"].to_dict()
    valid_team_codes = set(map(str, team_codes))

    for player_id in pitcher_ids_df["player_id"].astype(str).tolist():
        for year in years:
            try:
                day_df = fetch_player_day(player_id, year=year)
                if day_df.empty:
                    if verbose:
                        print(f"[playerDay] {player_id} year={year}: 0 rows")
                    continue

                if "team_code" in day_df.columns:
                    day_df["team_code"] = day_df["team_code"].astype(str)
                    day_df = day_df[day_df["team_code"].isin(valid_team_codes)].copy()

                if "game_date" in day_df.columns:
                    day_df = day_df[
                        (day_df["game_date"].dt.normalize() >= min_keep_date.normalize()) &
                        (day_df["game_date"].dt.normalize() <= max_date.normalize())
                    ].copy()

                if day_df.empty:
                    if verbose:
                        print(f"[playerDay] {player_id} year={year}: filtered to 0 rows")
                    continue

                if "player_name" not in day_df.columns:
                    day_df["player_name"] = pitcher_name_map.get(player_id, pd.NA)
                else:
                    day_df["player_name"] = day_df["player_name"].fillna(pitcher_name_map.get(player_id, pd.NA))

                if "GS" in day_df.columns and "is_starting_pitcher" not in day_df.columns:
                    day_df["is_starting_pitcher"] = (
                        pd.to_numeric(day_df["GS"], errors="coerce").fillna(0) > 0
                    ).astype(int)

                appearance_frames.append(day_df)
                if verbose:
                    print(f"[playerDay] {player_id} year={year}: {len(day_df)} rows")
            except Exception as e:
                print(f"[playerDay] {player_id} year={year}: ERROR -> {e}")
            time.sleep(sleep_sec)

    if not appearance_frames:
        return pd.DataFrame()

    appearance_df = pd.concat(appearance_frames, ignore_index=True).drop_duplicates()
    appearance_df = prepare_bullpen_appearance_df(
        appearance_df,
        game_id_col="game_id",
        game_date_col="game_date",
        team_code_col="team_code",
        player_id_col="player_id",
        player_name_col="player_name",
        ip_col="IP",
        np_col="NP",
        er_col="ER",
        bb_col="BB",
        hr_col="HR",
        h_col="H",
        k_col="K",
        is_starting_pitcher_col="is_starting_pitcher",
        gs_col="GS",
    )
    return appearance_df


def append_recent_bullpen_features_via_api(
    model_df: pd.DataFrame,
    *,
    roster_code: str = "1",
    lookback_days: int = 7,
    sleep_sec: float = 0.2,
    fip_constant: float = 3.2,
    game_id_col: str = "game_id",
    game_date_col: str = "game_date",
    home_team_code_col: str = "home_team_code",
    away_team_code_col: str = "away_team_code",
    season_col: Optional[str] = "season",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    model_df에서 필요한 날짜/팀을 추출해 appearance_df를 API로 조립하고,
    최근 불펜 피처를 붙여 반환한다.
    """
    if model_df.empty:
        return model_df.copy()

    if game_date_col not in model_df.columns:
        raise ValueError(f"model_df에 필요한 컬럼이 없습니다: {game_date_col}")

    date_list = (
        pd.to_datetime(model_df[game_date_col], errors="coerce")
        .dropna()
        .dt.strftime("%Y-%m-%d")
        .drop_duplicates()
        .tolist()
    )

    team_codes: List[str] = []
    if home_team_code_col in model_df.columns:
        team_codes.extend(model_df[home_team_code_col].dropna().astype(str).tolist())
    if away_team_code_col in model_df.columns:
        team_codes.extend(model_df[away_team_code_col].dropna().astype(str).tolist())
    team_codes = sorted(set(team_codes))

    appearance_df = collect_pitcher_appearance_df_by_dates(
        date_list,
        team_codes=team_codes,
        roster_code=roster_code,
        lookback_days=lookback_days,
        sleep_sec=sleep_sec,
        verbose=verbose,
    )

    return append_recent_bullpen_features(
        model_df,
        appearance_df,
        game_id_col=game_id_col,
        game_date_col=game_date_col,
        home_team_code_col=home_team_code_col,
        away_team_code_col=away_team_code_col,
        season_col=season_col,
        fip_constant=fip_constant,
    )
