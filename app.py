from __future__ import annotations

import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from urllib.error import HTTPError, URLError
import importlib

import pandas as pd
import streamlit as st

import prediction_engine as engine
engine = importlib.reload(engine)


st.set_page_config(
    page_title="Statiz 승부예측",
    page_icon="⚾",
    layout="wide",
)


MODEL_LABELS = {
    "elo_full": "elo_full",
    "elo_sp": "elo_sp",
    "elo_sp_l2": "elo_sp_l2",
}


def init_state() -> None:
    defaults = {
        "pred_df": None,
        "pred_df_core": None,
        "pred_df_elo": None,
        "result_df": None,
        "submit_df": None,
        "submit_result_df": None,
        "selected_model": None,
        "log_df": None,
        "games_with_elo": None,
        "updated_snapshot": None,
        "load_logs": "",
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def reset_downstream(from_step: str) -> None:
    order = {
        "load": ["pred_df_core", "pred_df_elo", "result_df", "submit_df", "submit_result_df", "selected_model", "log_df", "games_with_elo", "updated_snapshot", "load_logs"],
        "impute": ["pred_df_core", "pred_df_elo", "result_df", "submit_df", "submit_result_df", "selected_model", "log_df", "games_with_elo", "updated_snapshot"],
        "features": ["result_df", "submit_df", "submit_result_df", "selected_model", "log_df"],
        "predict": ["submit_result_df", "log_df"],
    }
    for key in order[from_step]:
        st.session_state[key] = None


def show_status_card(title: str, body: str, state: str = "ready") -> None:
    color = {
        "ready": "#f4f7fb",
        "done": "#edf8f2",
        "warn": "#fff7e6",
        "error": "#fff0f0",
    }[state]
    st.markdown(
        f"""
        <div style="background:{color}; border:1px solid #d7dee8; border-radius:8px; padding:14px 16px; margin:8px 0 14px 0;">
            <div style="font-weight:700; font-size:15px;">{title}</div>
            <div style="font-size:13px; color:#475569; margin-top:4px;">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def display_result_table(df: pd.DataFrame) -> None:
    cols = [
        "game_id",
        "game_date",
        "home_team",
        "away_team",
        "home_team_code",
        "away_team_code",
        "home_win_percent",
        "away_win_percent",
        "model_name",
    ]
    show_cols = [c for c in cols if c in df.columns]
    st.dataframe(df[show_cols], use_container_width=True, hide_index=True)


def extract_http_status(exc: BaseException) -> int | None:
    current: BaseException | None = exc
    seen = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, HTTPError):
            return int(current.code)
        for attr in ("code", "status", "status_code"):
            value = getattr(current, attr, None)
            if isinstance(value, int):
                return value
        current = current.__cause__ or current.__context__
    return None


def show_error(title: str, exc: BaseException) -> None:
    status_code = extract_http_status(exc)
    if status_code is not None:
        st.error(f"{title} (HTTP {status_code})")
    elif isinstance(exc, URLError):
        st.error(f"{title} (URL error)")
    else:
        st.error(title)
    st.code("".join(traceback.format_exception_only(type(exc), exc)).strip(), language="text")


def is_empty_schedule_error(exc: BaseException) -> bool:
    message = str(exc)
    return "일정이 비어 있습니다" in message or "schedule" in message.lower() and "empty" in message.lower()


def show_load_exception(date_str: str, exc: BaseException) -> None:
    if is_empty_schedule_error(exc):
        st.warning(f"{date_str} 경기 일정이 비어 있습니다.")
        st.info(
            "API 인증 실패라기보다 선택한 날짜에 경기 일정이 없거나 아직 Statiz API에 일정이 내려오지 않은 경우입니다. "
            "다른 날짜를 선택해서 다시 불러와 주세요."
        )
        st.caption("HTTP status: 없음 - API 호출 이후 빈 일정으로 판단된 오류입니다.")
        st.code("".join(traceback.format_exception_only(type(exc), exc)).strip(), language="text")
        return

    show_error("데이터 불러오기 실패", exc)


def run_with_elapsed(label: str, func, *args, **kwargs):
    timer_box = st.empty()
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        timer_box.info(f"{label} running... elapsed 0.0 sec")
        while not future.done():
            elapsed = time.perf_counter() - start
            timer_box.info(f"{label} running... elapsed {elapsed:,.1f} sec")
            time.sleep(0.2)
        elapsed = time.perf_counter() - start
        result = future.result()
    timer_box.success(f"{label} complete - total {elapsed:,.1f} sec")
    return result


init_state()

st.title("Statiz 승부예측 UI")
st.caption("prediction.ipynb 흐름을 단계별로 실행하는 로컬 웹 도구입니다.")

with st.sidebar:
    st.header("설정")
    selected_date = st.date_input("불러올 경기 날짜", value=date.today())
    date_str = selected_date.strftime("%Y-%m-%d")
    api_sleep_sec = st.number_input("API 호출 간격", min_value=0.05, max_value=1.00, value=0.12, step=0.01)
    recent_sleep_sec = st.number_input("최근 흐름 호출 간격", min_value=0.05, max_value=1.00, value=0.15, step=0.01)
    submit_sleep_sec = st.number_input("제출 호출 간격", min_value=0.05, max_value=1.00, value=0.20, step=0.01)
    verbose = st.toggle("API 상세 로그", value=False)

    st.divider()
    st.write("현재 선택 날짜")
    st.subheader(date_str)

    with st.expander("API key check", expanded=False):
        api_info = engine.get_api_debug_info()
        st.caption("값은 보안상 일부만 표시합니다. 노트북과 같은 utils.py / submit.py에서 로드되는지 확인하세요.")
        st.json(api_info)


left, right = st.columns([1.05, 0.95], gap="large")

with left:
    st.subheader("1. 경기 데이터 불러오기")
    show_status_card(
        "날짜를 선택하고 데이터를 불러오세요",
        "Statiz API와 라인업/선수 시즌 데이터를 함께 호출하므로 시간이 걸릴 수 있습니다.",
    )

    if st.button("경기 데이터 불러오기", type="primary", use_container_width=True):
        try:
            reset_downstream("load")
            with st.status("경기 데이터와 라인업 정보를 불러오는 중입니다...", expanded=True) as status:
                st.write("경기 일정 확인")
                st.write("라인업 및 선수 시즌 정보 수집")
                pred_df = run_with_elapsed(
                    "경기 데이터 불러오기",
                    engine.load_prediction_data_with_logs,
                    date_str=date_str,
                    sleep_sec=float(api_sleep_sec),
                    verbose=verbose,
                )
                pred_df, load_logs = pred_df
                st.session_state.pred_df = pred_df
                st.session_state.load_logs = load_logs
                status.update(label="데이터 불러오기 완료", state="complete", expanded=False)
            st.success(f"{len(st.session_state.pred_df)}개 경기 데이터를 불러왔습니다.")
        except Exception as exc:
            st.session_state.load_logs = getattr(exc, "logs", "")
            show_load_exception(date_str, exc)

    if st.session_state.load_logs:
        with st.expander("Notebook-style execution log", expanded=True):
            st.code(st.session_state.load_logs, language="text")

    if st.session_state.pred_df is not None:
        pred_df = st.session_state.pred_df
        missing_count = engine.missing_value_count(pred_df)
        st.metric("pred_df 결측치 개수", missing_count)
        st.caption("prediction.ipynb 7번 셀의 `pred_df.isna().sum().sum()` 결과입니다.")
        st.dataframe(
            pred_df[["game_id", "game_date", "home_team_code", "away_team_code"]].head(20),
            use_container_width=True,
            hide_index=True,
        )

    st.divider()
    st.subheader("2. 선발투수 결측치 보간")
    if st.session_state.pred_df is None:
        show_status_card("먼저 데이터를 불러와야 합니다", "경기 데이터가 준비되면 보간 실행 여부를 선택할 수 있습니다.", "warn")
    else:
        run_impute = st.radio(
            "선발투수 결측치 보간을 실행할까요?",
            options=["실행", "건너뛰기"],
            horizontal=True,
            index=0,
        )
        if st.button("보간 단계 확정", use_container_width=True):
            try:
                reset_downstream("impute")
                if run_impute == "실행":
                    with st.spinner("선발투수 결측치를 보간하고 관련 feature를 다시 계산하는 중입니다..."):
                        target_year = pd.to_datetime(date_str).year
                        st.session_state.pred_df = run_with_elapsed(
                            "선발투수 결측치 보간",
                            engine.fill_starting_pitcher_features,
                            st.session_state.pred_df,
                            target_year,
                        )
                    st.success("선발투수 결측치 보간을 완료했습니다.")
                else:
                    st.info("보간을 건너뛰고 현재 pred_df를 그대로 사용합니다.")
                st.metric("보간 후 결측치 개수", engine.missing_value_count(st.session_state.pred_df))
            except Exception as exc:
                show_error("보간 단계 실패", exc)

with right:
    st.subheader("3. 최근 흐름 + ELO 업데이트")
    if st.session_state.pred_df is None:
        show_status_card("대기 중", "경기 데이터와 보간 단계가 끝나면 실행할 수 있습니다.", "warn")
    else:
        st.caption("prediction.ipynb 16번, 18번 셀 흐름입니다. ELO snapshot은 로컬 파일에 항상 업데이트합니다.")
        if st.button("최근 흐름 및 ELO 반영", use_container_width=True):
            try:
                reset_downstream("features")
                with st.status("최근 경기 흐름과 ELO 정보를 반영하는 중입니다...", expanded=True) as status:
                    st.write("최근 14일 경기 흐름 수집")
                    st.write("team_elo_snapshot_2026.csv 업데이트")
                    pred_df_core, pred_df_elo, games_with_elo, updated_snapshot = run_with_elapsed(
                        "최근 흐름 및 ELO 반영",
                        engine.add_recent_and_elo_features,
                        st.session_state.pred_df,
                        True,
                        float(recent_sleep_sec),
                        float(submit_sleep_sec),
                    )
                    st.session_state.pred_df_core = pred_df_core
                    st.session_state.pred_df_elo = pred_df_elo
                    st.session_state.games_with_elo = games_with_elo
                    st.session_state.updated_snapshot = updated_snapshot
                    status.update(label="최근 흐름 및 ELO 반영 완료", state="complete", expanded=False)
                st.success("로컬 ELO snapshot 확인 및 feature merge를 완료했습니다.")
            except Exception as exc:
                show_error("최근 흐름/ELO 단계 실패", exc)

    if st.session_state.pred_df_elo is not None:
        pred_df_elo = st.session_state.pred_df_elo
        elo_cols = ["game_id", "home_team_code", "away_team_code", "home_elo_pre", "away_elo_pre", "elo_diff", "elo_exp_home"]
        st.dataframe(pred_df_elo[[c for c in elo_cols if c in pred_df_elo.columns]], use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("4. 모델 선택 및 예측")
    selected_model = st.selectbox(
        "사용할 모델",
        options=list(MODEL_LABELS.keys()),
        format_func=lambda key: MODEL_LABELS[key],
        index=2,
    )
    if st.session_state.pred_df_elo is None:
        show_status_card("예측 준비 전", "최근 흐름과 ELO feature를 먼저 반영해주세요.", "warn")
    else:
        if st.button("예측 실행 및 엑셀 로그 업데이트", type="primary", use_container_width=True):
            try:
                reset_downstream("predict")
                with st.status("모델 예측을 실행하는 중입니다...", expanded=True) as status:
                    st.write(f"{selected_model} 모델로 홈팀 승리 확률 계산")
                    result_df = run_with_elapsed(
                        "모델 예측",
                        engine.run_prediction,
                        st.session_state.pred_df_elo,
                        selected_model,
                    )
                    submit_df = engine.make_submit_df(result_df)
                    st.write("prediction_log.xlsx 업데이트")
                    log_df = run_with_elapsed(
                        "엑셀 로그 업데이트",
                        engine.save_prediction_log,
                        result_df,
                        selected_model,
                    )
                    st.session_state.result_df = result_df
                    st.session_state.submit_df = submit_df
                    st.session_state.log_df = log_df
                    st.session_state.selected_model = selected_model
                    status.update(label="예측 및 로그 업데이트 완료", state="complete", expanded=False)
                st.success("예측 결과를 만들고 로컬 엑셀 로그를 업데이트했습니다.")
            except Exception as exc:
                show_error("예측 실패", exc)

if st.session_state.result_df is not None:
    st.divider()
    st.subheader("예측 결과")
    display_result_table(st.session_state.result_df)

    st.subheader("제출용 데이터")
    st.dataframe(st.session_state.submit_df, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("5. API Submit")
    st.caption("prediction.ipynb 31번 셀의 출력과 같은 제출 결과를 화면에 표시합니다.")
    confirm_submit = st.checkbox("실제 제출 API를 실행하겠습니다.")
    if st.button("API 제출 실행", disabled=not confirm_submit, use_container_width=True):
        try:
            with st.status("Statiz API에 예측값을 제출하는 중입니다...", expanded=True) as status:
                submit_result_df = run_with_elapsed(
                    "API 제출",
                    engine.submit_predictions,
                    st.session_state.submit_df,
                    float(submit_sleep_sec),
                    True,
                )
                st.session_state.submit_result_df = submit_result_df
                status.update(label="API 제출 완료", state="complete", expanded=False)
            st.success("제출 API 실행이 끝났습니다.")
        except Exception as exc:
            show_error("API 제출 실패", exc)

if st.session_state.submit_result_df is not None:
    st.subheader("API 제출 결과")
    st.dataframe(st.session_state.submit_result_df, use_container_width=True, hide_index=True)
