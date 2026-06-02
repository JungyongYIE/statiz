from pathlib import Path

import pandas as pd
from openpyxl import load_workbook


LOG_COLUMNS = [
    "game_id",
    "game_date",
    "home_team",
    "away_team",
    "home_team_code",
    "away_team_code",
    "home_win_percent",
    "home_win_proba",
    "model_name",
    "result",
    "correct",
]

UPSERT_COLUMNS = [
    "game_id",
    "game_date",
    "home_team",
    "away_team",
    "home_team_code",
    "away_team_code",
    "home_win_percent",
    "home_win_proba",
    "model_name",
]

TEAM_CODE_TO_NAME = {
    1001: "\uC0BC\uC131",
    2002: "KIA",
    3001: "\uB86F\uB370",
    5002: "LG",
    6002: "\uB450\uC0B0",
    7002: "\uD55C\uD654",
    9002: "SSG",
    10001: "\uD0A4\uC6C0",
    11001: "NC",
    12001: "KT",
}


def _validate_required_columns(df: pd.DataFrame, required_cols: list[str]) -> None:
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _coerce_int_series(series: pd.Series, col_name: str) -> pd.Series:
    coerced = pd.to_numeric(series, errors="coerce")
    if coerced.isna().any():
        bad_rows = series[coerced.isna()].tolist()
        raise ValueError(f"`{col_name}` contains non-numeric values: {bad_rows}")
    return coerced.astype(int)


def _prepare_new_log_rows(pred_df: pd.DataFrame, model_name: str = "") -> pd.DataFrame:
    required_cols = [
        "game_id",
        "game_date",
        "home_team_code",
        "away_team_code",
        "home_win_proba",
    ]
    _validate_required_columns(pred_df, required_cols)

    df = pred_df.copy()
    df = df[required_cols]

    df["game_id"] = _coerce_int_series(df["game_id"], "game_id")
    df["home_team_code"] = _coerce_int_series(df["home_team_code"], "home_team_code")
    df["away_team_code"] = _coerce_int_series(df["away_team_code"], "away_team_code")
    game_date = pd.to_datetime(df["game_date"], errors="coerce")
    if game_date.isna().any():
        raise ValueError("`game_date` contains invalid datetime values.")
    df["game_date"] = game_date.dt.strftime("%Y-%m-%d")

    df["home_win_proba"] = pd.to_numeric(df["home_win_proba"], errors="coerce")
    if df["home_win_proba"].isna().any():
        raise ValueError("`home_win_proba` contains non-numeric values.")

    df["home_team"] = df["home_team_code"].map(TEAM_CODE_TO_NAME).fillna("")
    df["away_team"] = df["away_team_code"].map(TEAM_CODE_TO_NAME).fillna("")
    df["home_win_percent"] = (df["home_win_proba"] * 100).round(2)
    df["model_name"] = model_name
    df["result"] = pd.NA
    df["correct"] = pd.NA

    return df[LOG_COLUMNS]


def _load_existing_log(file_path: Path, sheet_name: str) -> pd.DataFrame:
    if not file_path.exists():
        return pd.DataFrame(columns=LOG_COLUMNS)

    old = pd.read_excel(file_path, sheet_name=sheet_name)
    for col in LOG_COLUMNS:
        if col not in old.columns:
            old[col] = pd.NA

    old = old[LOG_COLUMNS].copy()
    old["game_date"] = pd.to_datetime(old["game_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    return old


def _append_rows_to_existing_sheet(file_path: Path, sheet_name: str, rows: list[list]) -> None:
    wb = load_workbook(file_path)
    ws = wb[sheet_name]

    next_row = ws.max_row + 1
    for row_idx, row_values in enumerate(rows, start=next_row):
        for col_idx, value in enumerate(row_values, start=1):
            ws.cell(row=row_idx, column=col_idx, value=value)

    ws.freeze_panes = "A2"
    ws.auto_filter.ref = f"A1:K{ws.max_row}"
    wb.save(file_path)
    wb.close()


def _upsert_rows_to_existing_sheet(file_path: Path, sheet_name: str, df: pd.DataFrame) -> None:
    wb = load_workbook(file_path)
    ws = wb[sheet_name]

    existing_row_by_game_id = {}
    for row_idx in range(2, ws.max_row + 1):
        game_id_value = ws.cell(row=row_idx, column=1).value
        if game_id_value in (None, ""):
            continue
        try:
            existing_row_by_game_id[int(game_id_value)] = row_idx
        except (TypeError, ValueError):
            continue

    append_start_row = ws.max_row + 1
    append_offset = 0

    for _, row in df.iterrows():
        game_id = int(row["game_id"])
        target_row = existing_row_by_game_id.get(game_id)

        if target_row is None:
            target_row = append_start_row + append_offset
            append_offset += 1
            existing_row_by_game_id[game_id] = target_row

        for col_idx, col_name in enumerate(UPSERT_COLUMNS, start=1):
            value = row[col_name]
            if pd.isna(value):
                value = None
            ws.cell(row=target_row, column=col_idx, value=value)

    ws.freeze_panes = "A2"
    ws.auto_filter.ref = f"A1:K{ws.max_row}"
    wb.save(file_path)
    wb.close()


def append_prediction_log(
    pred_df: pd.DataFrame,
    file_path: str = "prediction_log.xlsx",
    sheet_name: str = "Raw_Submission_Log",
    dedup: bool = True,
    model_name: str = "",
) -> pd.DataFrame:
    """
    Append prediction rows to the bottom of prediction_log.xlsx.

    Required input columns:
    game_id, game_date, home_team_code, away_team_code, home_win_proba
    """

    path = Path(file_path)
    new_rows = _prepare_new_log_rows(pred_df=pred_df, model_name=model_name)
    old_rows = _load_existing_log(file_path=path, sheet_name=sheet_name) if path.exists() else pd.DataFrame(columns=LOG_COLUMNS)

    if path.exists():
        if dedup:
            _upsert_rows_to_existing_sheet(
                file_path=path,
                sheet_name=sheet_name,
                df=new_rows,
            )
            existing_without_duplicates = old_rows.copy()
            existing_without_duplicates["game_id"] = pd.to_numeric(
                existing_without_duplicates["game_id"], errors="coerce"
            )
            existing_without_duplicates = existing_without_duplicates[
                ~existing_without_duplicates["game_id"].isin(new_rows["game_id"])
            ]
            out = pd.concat([existing_without_duplicates, new_rows], ignore_index=True)
        else:
            if not new_rows.empty:
                _append_rows_to_existing_sheet(
                    file_path=path,
                    sheet_name=sheet_name,
                    rows=new_rows.where(pd.notna(new_rows), None).values.tolist(),
                )
            out = pd.concat([old_rows, new_rows], ignore_index=True)
    else:
        out = new_rows.copy()
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            out.to_excel(writer, sheet_name=sheet_name, index=False)
            ws = writer.sheets[sheet_name]
            ws.freeze_panes = "A2"
            ws.auto_filter.ref = ws.dimensions

    print(f"Saved log: {path}")
    print(f"Total rows: {len(out)}")

    return out
