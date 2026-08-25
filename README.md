# Statiz Prediction

Statiz 승부예측 대회용 KBO 경기 결과 예측 프로젝트입니다. Statiz API에서 경기 일정, 라인업, 선수 시즌 성적을 수집하고, 타자/선발투수/최근 흐름/ELO feature를 만든 뒤 로지스틱 회귀 기반 모델로 홈팀 승리 확률을 예측합니다. 예측 결과는 Streamlit UI 또는 notebook에서 확인하고, 로컬 제출 로그 저장 및 Statiz 제출 API 호출까지 이어집니다.

## 전체 흐름

```text
Statiz API
  -> utils.py / API_load.py
  -> model_df_2023.csv, model_df_2024.csv, model_df_2025.csv
  -> modeling_*.ipynb
  -> logit_model_*.pkl + feature_cols_*.pkl
  -> prediction_engine.py
  -> app.py 또는 prediction.ipynb
  -> prediction_log.xlsx
  -> submit.py
```

운영 예측 기준 흐름은 다음과 같습니다.

1. `utils.build_prediction_data_for_date()`가 선택 날짜의 경기, 라인업, 로스터, 선수 시즌 성적을 API로 수집합니다.
2. `utils`의 전처리 함수들이 경기 단위 `pred_df`를 생성합니다.
3. `prediction_engine.fill_starting_pitcher_features()`가 선발투수 결측 feature를 보정합니다.
4. `utils.add_recent_flow_features_for_prediction()`와 `elo.merge_elo()`가 최근 흐름 및 ELO feature를 붙입니다.
5. `prediction_engine.run_prediction()`이 선택 모델로 홈팀 승리 확률을 계산합니다.
6. `save_log.append_prediction_log()`가 `prediction_log.xlsx`에 예측 기록을 남깁니다.
7. `submit.submit_predictions_df()`가 제출 API로 예측값을 전송합니다.

## 주요 파일

| 파일 | 역할 |
| --- | --- |
| `app.py` | Streamlit 기반 실행 UI입니다. 날짜 선택, API 로드, 결측 보정, 최근 흐름/ELO 병합, 모델 예측, 로그 저장, 제출 API 호출을 단계별 버튼으로 실행합니다. |
| `prediction_engine.py` | 앱과 notebook에서 공통으로 쓰는 예측 오케스트레이션 모듈입니다. 데이터 로드, feature 보정, ELO 병합, 모델 로드/학습, 제출용 dataframe 생성, 로그 저장, API 제출을 하나의 인터페이스로 묶습니다. |
| `prediction.ipynb` | 실제 예측 절차를 notebook으로 실행하던 작업 파일입니다. `app.py`는 이 notebook의 실행 흐름을 UI로 옮긴 형태입니다. |
| `utils.py` | Statiz API GET 호출, 경기/라인업/로스터/선수 시즌 성적 수집, 전처리, 타자/선발투수 feature 생성, 경기 단위 modeling/prediction dataframe 생성을 담당하는 핵심 유틸 모듈입니다. |
| `API_load.py` | API 기반 추가 feature 수집 모듈입니다. 특히 bullpen appearance 전처리, 최근 bullpen 품질/피로도 feature 생성, 날짜별 bullpen feature 수집 및 병합 로직을 포함합니다. |
| `elo.py` | 팀 ELO snapshot 업데이트 및 예측 dataframe에 ELO feature를 병합합니다. 기본 snapshot 파일은 `team_elo_snapshot_2026.csv`입니다. |
| `save_log.py` | 예측 결과를 `prediction_log.xlsx`의 `Raw_Submission_Log` sheet에 저장하거나 같은 `game_id` 기준으로 upsert합니다. |
| `submit.py` | Statiz 제출 API 호출 모듈입니다. `game_id`와 홈팀 승리 확률 퍼센트를 받아 `prediction/savePrediction` 또는 지정 path로 POST 요청을 보냅니다. |
| `함수모음.ipynb` | 실험 과정에서 사용한 함수와 초기 탐색 코드 모음입니다. 현재 운영 흐름에서는 재사용 가능한 로직이 주로 `utils.py`, `API_load.py`, `elo.py`로 분리되어 있습니다. |

## 모델링 파일

| 파일 | 설명 |
| --- | --- |
| `modeling_1st.ipynb` | 초기 feature set으로 로지스틱 회귀 모델을 학습하고 `logit_model.pkl`, `feature_cols.pkl`을 저장한 실험입니다. |
| `modeling_sp.ipynb` | 선발투수 feature 중심 실험입니다. `logit_model_sp.pkl`, `feature_cols_sp.pkl`을 생성합니다. |
| `modeling_2nd_recent.ipynb` | 최근 경기 흐름 feature를 추가한 실험입니다. `recent`, `reduced_core`, `reduce_core_sp` 계열 모델/feature artifact를 생성합니다. |
| `modeling_3rd_bullpen.ipynb` | bullpen 관련 feature를 실험한 notebook입니다. |
| `modeling_4th_elo.ipynb` | ELO feature를 결합한 운영 후보 모델 실험입니다. `logit_model_elo_full.pkl`, `feature_cols_elo_full.pkl`, `logit_model_elo_sp.pkl`, `feature_cols_elo_sp.pkl`을 생성합니다. |
| `modeling_5th_experiment.ipynb` | `elo_sp` feature set을 기준으로 logistic baseline, XGBoost, LightGBM, MLP, ensemble 계열을 비교한 추가 실험입니다. 결과는 `modeling_5th_outputs/`에 저장됩니다. |
| `modeling_non.ipynb` | 선발투수 feature를 제외하거나 다른 feature 조합을 비교한 보조 실험입니다. |

## 모델 및 feature artifact

저장된 `.pkl` 파일은 `joblib`으로 로드되는 모델 또는 feature column 목록입니다.

| Artifact | 용도 |
| --- | --- |
| `logit_model.pkl`, `feature_cols.pkl` | 초기 전체 feature 로지스틱 모델 |
| `logit_model_sp.pkl`, `feature_cols_sp.pkl` | 선발투수 중심 로지스틱 모델 |
| `logit_model_recent.pkl`, `feature_cols_recent.pkl` | 최근 흐름 feature 포함 모델 |
| `logit_model_reduced_core.pkl`, `feature_cols_reduced_core.pkl` | 축소 core feature 모델 |
| `logit_model_reduce_core_sp.pkl`, `feature_cols_recent_reduce_core_sp.pkl` | 축소 core + 선발투수 feature 모델 |
| `logit_model_elo_full.pkl`, `feature_cols_elo_full.pkl` | ELO 포함 full 운영 후보 모델 |
| `logit_model_elo_sp.pkl`, `feature_cols_elo_sp.pkl` | ELO + 선발투수 중심 운영 후보 모델 |
| `modeling_5th_outputs/modeling_5th_elo_sp_feature_cols.pkl` | 5차 실험에서 사용한 `elo_sp` feature column 목록 |

현재 `prediction_engine.py`에서 직접 선택 가능한 모델명은 다음과 같습니다.

| 모델명 | 방식 |
| --- | --- |
| `elo_full` | `logit_model_elo_full.pkl`과 `feature_cols_elo_full.pkl`을 로드해 예측 |
| `elo_sp` | `logit_model_elo_sp.pkl`과 `feature_cols_elo_sp.pkl`을 로드해 예측 |
| `elo_sp_l2` | `feature_cols_elo_sp.pkl`을 사용하고, `model_df_2023.csv`~`model_df_2025.csv`로 L2 로지스틱 회귀를 즉시 재학습해 예측 |

## 데이터 파일

| 파일 | 설명 |
| --- | --- |
| `model_df_2023.csv`, `model_df_2024.csv`, `model_df_2025.csv` | 모델 학습 및 재학습에 사용하는 연도별 경기 단위 데이터입니다. |
| `games_2026_with_elo.csv` | 2026 경기와 ELO feature를 결합한 중간 산출물입니다. |
| `team_elo_snapshot_2026.csv` | 운영 예측에서 사용하는 최신 팀 ELO snapshot입니다. `elo.update_team_elo_snapshot_2026()` 실행 시 업데이트됩니다. |
| `team_elo_snapshot.csv` | 이전 또는 보조 ELO snapshot입니다. |
| `seaon_pit_df.csv`, `sp_df.csv` | 선발투수 결측 보정 및 feature 확인에 쓰이는 투수 관련 보조 데이터입니다. |
| `ex_pred_df.csv`, `x_pred.csv` | 예측 입력/결과 확인용 예시 파일입니다. |
| `prediction_log.xlsx` | 제출 전후 예측값을 누적 저장하는 운영 로그입니다. |
| `statiz_prediction_v*.xlsx`, `statiz_submission_log_from_screenshots.xlsx` | 과거 제출/정리 산출물입니다. |
| `team_*.json` | 팀별 API 응답 또는 로컬 참고 데이터입니다. |

## 실행 방법

### 1. 환경 준비

```powershell
pip install -r requirements.txt
```

Anaconda Python을 사용하는 경우 `run_app_anaconda.bat`를 실행하면 현재 폴더에서 Streamlit 앱이 바로 실행됩니다.

```powershell
.\run_app_anaconda.bat
```

직접 실행하려면 다음 명령을 사용합니다.

```powershell
streamlit run app.py --server.address 127.0.0.1 --server.port 8501
```

### 2. Streamlit UI 예측

1. 사이드바에서 예측할 경기 날짜를 선택합니다.
2. `경기 데이터 불러오기` 단계에서 Statiz API 데이터를 수집합니다.
3. 선발투수 결측 보정 여부를 선택합니다.
4. 최근 흐름과 ELO feature를 반영합니다.
5. `elo_full`, `elo_sp`, `elo_sp_l2` 중 모델을 선택해 예측합니다.
6. 결과가 `prediction_log.xlsx`에 저장되는지 확인합니다.
7. 실제 제출이 필요할 때만 제출 확인 checkbox를 켠 뒤 API Submit을 실행합니다.

### 3. Notebook 예측

UI 대신 notebook에서 실행하려면 `prediction.ipynb`를 사용합니다. notebook의 흐름은 `prediction_engine.py`에 정리된 함수들과 대응됩니다.

## 제출 포맷

`prediction_engine.make_submit_df()`는 제출용 dataframe을 다음 형태로 만듭니다.

| column | 의미 |
| --- | --- |
| `game_id` | Statiz 경기 번호 |
| `home_win_proba` | 홈팀 승리 확률을 퍼센트 단위로 변환한 값 |

`submit.submit_predictions_df()`는 각 row를 순회하며 제출 API에 `s_no`, `percent`를 보냅니다. API 과호출을 피하기 위해 `sleep_sec` 간격을 둡니다.

## 로그 관리

`save_log.py`는 다음 column을 기준으로 예측 로그를 관리합니다.

```text
game_id, game_date, home_team, away_team,
home_team_code, away_team_code,
home_win_percent, home_win_proba,
model_name, result, correct
```

기본 설정에서는 `game_id`가 이미 존재하면 기존 row를 업데이트하고, 없으면 새 row를 추가합니다. 실제 경기 결과가 나온 뒤 `result`, `correct` column을 채우면 제출 성과 추적에 사용할 수 있습니다.

## 5차 실험 결과 요약

`modeling_5th_outputs/modeling_5th_results.csv` 기준으로 `elo_sp` feature set에서 여러 모델을 비교했습니다.

| metric | best model | score |
| --- | --- | --- |
| Accuracy | `logistic_baseline` | 0.6262 |
| AUC | `ensemble_soft_voting` | 0.6550 |
| Brier score | `ensemble_soft_voting` | 0.2319 |

이 결과는 `modeling_5th_experiment.ipynb`의 실험 산출물이며, 현재 앱의 기본 선택지는 `elo_sp_l2`입니다.

## 주의사항

- `utils.py`, `API_load.py`, `submit.py`에는 Statiz API 인증 및 서명 로직이 포함되어 있습니다. 공개 저장소로 배포할 경우 API key/secret은 환경변수 또는 별도 설정 파일로 분리하는 것을 권장합니다.
- `team_elo_snapshot_2026.csv`는 운영 중 업데이트되는 파일입니다. 예측 전 snapshot 기준일이 최신인지 확인해야 합니다.
- `prediction_log.xlsx`는 제출 기록이 누적되는 파일입니다. 같은 `game_id`는 기본적으로 upsert됩니다.
- Statiz API 응답 구조나 경기 상태 코드가 바뀌면 `utils.fetch_game_schedule()`, `submit.submit_one_prediction()`의 path/parameter/signature 규칙을 먼저 확인해야 합니다.
- 일부 notebook과 과거 스크립트는 실험 과정의 산출물입니다. 운영 예측은 `app.py`, `prediction_engine.py`, `utils.py`, `elo.py`, `save_log.py`, `submit.py`를 중심으로 확인하는 것이 좋습니다.
