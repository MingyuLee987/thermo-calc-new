# Ce–Cr 내식강: Thermo-Calc + 실험 기반 Active Learning

발표자료 `AI 활용방안.pptx` 8–17쪽, 제공한 공정 조건, 기존 저장소를 바탕으로 작성한 별도 실행 프로젝트입니다. 기존 GitHub 파일은 변경하지 않았습니다. 실제 Thermo-Calc 계산 결과와 실제 SP-240 측정값만 연구 데이터로 사용합니다.

## 1. 반영한 공정과 계산의 의미

| 공정 | 반영 방법 |
|---|---|
| Indutherm VTC-200V 진공/가압 주조 | 공정 메타데이터; Scheil은 장비의 유동·열전달을 모사하지 않음 |
| 약 1000°C 열간압연, 최종 판재 3 mm, 이후 공랭 | 메타데이터 및 동일 실험 조건 구분 |
| 시편 30×10×3 mm | 메타데이터; 전극 노출 면적과 같다고 가정하지 않음 |
| 900°C, 15분, Ar 분위기 | 평형 계산 온도 900°C; 유지시간/분위기는 메타데이터 |
| 이후 수냉, 원문에서 베이나이트 형성으로 기술 | 원문 기록; 베이나이트 분율을 평형 계산으로 예측하지 않음 |
| Ce 순도 99.9%, RND KOREA | 메타데이터; Ce 회수율이나 용해 손실은 자동 보정하지 않음 |

**900°C의 FCC_A1 분율은 열처리 온도에서의 평형 지표입니다. 최종 수냉 조직의 베이나이트 분율이 아닙니다.** 15분 안에 평형에 도달하는지, 수냉 중 베이나이트/마르텐사이트가 어떻게 형성되는지는 별도의 확산·변태 해석 또는 CCT/조직 실험이 필요합니다. 이 코드에는 해당 속도론 계산을 넣지 않았습니다.

제공된 잉곳 질량 600 g과 치수 150×100×180 mm는 함께 사용하면 밀도가 약 0.222 g/cm³가 되므로 강재 잉곳의 정보로 서로 맞지 않습니다. 원문 숫자를 `settings.json`에 보존하고 계산에는 사용하지 않았습니다.

## 2. 구성과 기존 코드 대비 변경

- `pipeline.py`: 단계별 실행, 입력 확인, 중간 저장 및 재시작
- `tc_backend.py`: 실제 TC-Python 평형/Scheil 연동
- `learning.py`: GPR 커널 비교, EI, 다음 실험 후보와 예측 지도
- `core.py`: 조성 생성, 단위/CSV 검사, Scheil 지표, 저장
- `settings.json`: 연구 조건과 공정 기록
- `test_pipeline.py`: Thermo-Calc 없이 실행하는 회귀 검증

기존 `01_tc_calphad_screen1.py`, `02_tc_scheil_selected.py`, `04_active_learning.py`의 흐름을 이어받되 다음을 수정했습니다.

1. Scheil에 `CompositionUnit.MASS_PERCENT`를 지정합니다. 공식 API의 기본 조성 단위는 mol%이므로 wt%를 그대로 넣으면 의도와 다른 조성이 됩니다.
2. 응고 곡선을 냉각 방향으로 정렬해 고상 몰분율이 0.90/0.99 이상이 되는 첫 반환 지점의 온도를 읽습니다. T99에 도달하지 않으면 NaN으로 남깁니다. 보간값이 아닌 반환 지점 기준입니다.
3. 반환 최고/최저 온도를 액상선/고상선이라고 단정하지 않습니다. 출력 이름을 `scheil_start_returned_C`, `scheil_end_C`, `calculated_temperature_span_C`로 구분합니다.
4. `FCC_A1#1`, `FCC_A1#2` 등 실제 상 인스턴스를 조회한 다음 합산합니다. 여러 FCC 조성집합의 Cr 농도는 상의 질량분율로 가중 평균합니다.
5. 상 질의 오류를 0으로 대체해 통과시키지 않습니다. 계산 실패는 필터에서 제외하고 오류 내용을 보존합니다.
6. SQLite에 조성별로 저장합니다. 재실행 시 완료된 점을 건너뛰고 설정이 달라지면 같은 결과 폴더를 재사용하지 못하게 합니다.
7. ARD Matérn 5/2, RBF, Matérn 3/2를 조성별 Leave-One-Out 검증으로 비교합니다. NLPD를 우선하고 RMSE와 95% 예측구간 포함률도 저장합니다.
8. 측정 SD와 반복 수로 평균의 표준오차를 구하고, 출력 정규화에 맞춰 관측오차 분산도 정규화합니다. 불확실도에는 별도 WhiteKernel 관측잡음을 중복 추가하지 않습니다.
9. A는 log10(i_corr)의 EI, B는 −E_pit의 EI, C는 불확실도와 CALPHAD 변화 지표를 사용합니다. 이미 측정한 조성과 A/B/C 중복을 제외합니다.
10. 기존 CSV 양식의 헤더/행 필드 수 불일치와 조성 중복을 검사합니다. 가짜 실험값이 포함된 양식을 배포하지 않습니다.

## 3. 설치

Thermo-Calc에서 사용하도록 구성한 Python 환경에서 이 프로젝트 폴더를 열고 실행합니다.

```powershell
python -m pip install -r requirements.txt
python pipeline.py check
```

`tc_python`은 일반 분석 패키지와 별개입니다. 사용 중인 Thermo-Calc 버전에 포함된 **TC-Python SDK 설치 안내**에 따라 설치하고 TC-Python/TCFE13 라이선스를 확인해야 합니다. 임의의 동명 패키지를 PyPI에서 설치하지 마세요. 설정은 기존 코드의 TCFE13을 유지했으며 실제 설치 버전은 아직 전달받지 못했습니다. `check`는 대표 조성 1개의 평형과 Scheil을 실제로 계산하므로 DB/원소/라이선스/API 연결을 함께 확인합니다.

현재 작성 환경에는 TC-Python이 없어 실제 엔진 실행은 검증하지 못했습니다. 공식 2026b API 문서를 대조했으나 사용자의 설치 버전과 DB에서 `check` 성공을 확인해야 합니다.

## 4. 실행 순서

### 평형 스크리닝

```powershell
# 먼저 5개 신규 점으로 연동 확인
python pipeline.py equilibrium --out results --limit 5
# 이어서 남은 전체 조성 계산; 이미 저장된 점은 건너뜀
python pipeline.py equilibrium --out results
# 실패한 조성 재시도
python pipeline.py equilibrium --out results --retry-failed
```

기본 그리드는 Ce=0.000–0.099 wt%, Cr=0.800–1.399 wt%, 모두 0.001 간격으로 **60,000개**입니다. Fe는 잔량이고 고정 원소는 발표자료의 C 0.050, Si 0.750, Mn 0.600, Cu 0.350, Ni 0.200, P 0.015, S 0.005 wt%입니다. 이는 실제 합금 분석값이 아니라 주어진 계산 설정입니다.

평형 필터는 FCC_A1 몰분율 ≥0.95, 지정 유해상 몰분율 합 ≤0.01, FCC 내부 Cr ≥0.80 wt%입니다. 해당 기준은 발표자료의 연구 설계값이며 범용 내식성 보증 기준이 아닙니다. 사용 DB의 실제 상 이름과 모델링 범위를 점검해야 합니다. 설정 유해상 목록에 없는 탄화물/Ce 화합물까지 자동으로 유해상에 포함하지 않습니다.

각 계산 직후 체크포인트를 저장하고 CSV는 단계 종료/정상 예외 종료 때 내보냅니다. 강제 종료되어도 다음 실행에서 SQLite를 읽어 이어갑니다. 서버가 복구 불가능한 오류를 낸 경우 세션을 중단하며 같은 명령을 재실행합니다. 동일 결과 폴더를 두 프로세스에서 동시에 사용하지 마세요.

### 초기 후보 10개와 응고 분석

```powershell
python pipeline.py initial --out results --n 10
python pipeline.py scheil --out results --candidates results/initial_candidates.csv
python pipeline.py template --out results
```

초기 후보는 Ce/Cr 범위를 정규화한 뒤 maximin 방식으로 떨어진 조성을 선택합니다. 성능 최적 후보라는 뜻은 아닙니다. `scheil_descriptors.csv`에서 T90, T99, ΔT90–99와 실패 상태를 보고 제조 후보를 검토하세요. `scheil_curves`에는 고상 몰분율 곡선과 별도 온도 격자의 잔류 액상 Ce/Cr 농도 곡선을 저장합니다. 액상 조성 질의가 지원되지 않으면 응고 결과를 보존하면서 `segregation_status`와 오류를 기록합니다. 마지막 반환 액상 조성은 측정된 편석값이나 완전 응고 후 전체 조성이 아닙니다.

기본값은 `max_dT90_99_C: null`입니다. 발표자료에 수치형 제조성 컷오프가 없으므로 임의의 합격 기준을 만들지 않았습니다. ΔT90–99는 비교 지표이며 균열 확률 또는 보장된 주조성으로 해석하지 않습니다. 이 상태에서 추천 후보는 **평형 필터만 통과**하며 신규 A/B/C에 대해서도 제조 전에 Scheil 결과를 검토해야 합니다.

제조성 기준을 연구적으로 정한 경우, 시작 전 별도의 설정 파일에서 `max_dT90_99_C`를 정하고 새 `--out` 폴더를 사용하세요. 해당 설정으로 평형 계산 후 아래처럼 전체 평형 통과 풀의 Scheil을 계산합니다.

```powershell
python pipeline.py equilibrium --config settings_manufacturing.json --out results_manufacturing
python pipeline.py scheil --config settings_manufacturing.json --out results_manufacturing
python pipeline.py initial --config settings_manufacturing.json --out results_manufacturing --n 10
```

수치 컷오프를 켜면 초기 후보와 학습 추천 모두 Scheil 성공 + T99 도달 + ΔT 컷오프 충족 후보만 허용합니다. 계산하지 않은 조성을 통과한 것으로 간주하지 않습니다. 필요하면 `scheil --limit 100`으로 나누어 계산할 수 있습니다.

### 실제 부식시험 데이터 입력

생성된 `results/sp240_results.csv`의 빈 칸에 실제 측정값을 입력합니다.

| 열 | 단위 및 의미 |
|---|---|
| Ce_wt, Cr_wt | wt%; 기본적으로 제조 목표 조성, 전 주기 동일 정의 유지 |
| icorr_mean_A_cm2, icorr_sd_A_cm2 | A/cm²; 반복 측정 평균과 표준편차 |
| Epit_mean_V_AgAgCl, Epit_sd_V | V vs Ag/AgCl; 평균과 표준편차 |
| Rct_mean_ohm_cm2, Rct_sd_ohm_cm2 | Ω·cm²; 면적 정규화된 값, 선택 입력 |
| Ecorr_mean_V_AgAgCl | 보조 기록; 기본 추천 목표는 아님 |
| n_replicates | 같은 조성의 반복 측정 횟수, 양의 정수 |
| experiment_context | 전해질·열처리·측정 기준; 설정과 같아야 함 |

`i_corr`에 μA/cm² 값을 넣으려면 10^-6을 곱합니다. Rct가 Ω라면 실제 노출 면적으로 Ω·cm²로 변환한 값을 사용합니다. 시편의 전체 표면적을 노출 면적으로 자동 가정하지 않습니다. 이 코드는 원시 분극/EIS 곡선을 자동 피팅하거나 SP-240 장비를 제어하지 않습니다. 검증된 피팅/실험 처리 결과를 입력받습니다.

조성당 한 행으로 반복을 요약합니다. 같은 조성을 다시 시험하면 누적 평균·SD·반복 수를 적절히 재계산하고 중복 행으로 추가하지 마세요. 최소 6개 서로 다른 조성의 i_corr/E_pit 데이터가 필요하며 초기 8–12개를 권장합니다. E_pit 미검출/측정한계 초과는 0이나 임의의 큰 값으로 대체하지 말아야 합니다. 현재 버전은 검열자료 모델을 구현하지 않았으므로 그런 데이터가 있다면 해당 모델을 추가해야 합니다. Rct는 전 행이 비어 있으면 생략되며 일부만 입력된 경우에는 입력 오류로 중단합니다.

### 모델 학습과 다음 A/B/C 추천

```powershell
python pipeline.py recommend --out results --experiments results/sp240_results.csv
python pipeline.py scheil --out results --candidates results/next_candidates.csv
python pipeline.py template --out results --candidates results/next_candidates.csv --template-file results/cycle02_blank.csv
```

새 후보를 제조·측정한 뒤 누적 `sp240_results.csv`에 실제 결과를 합치고 `recommend`를 다시 실행합니다. 각 주기의 추천/학습 보고서는 다음 실행 전에 별도 폴더에 보관하세요. `recommend` 출력 파일은 최신 주기로 갱신됩니다. 템플릿 명령은 기존 실험 CSV를 덮어쓰지 않습니다.

모든 목적을 최소화 형태로 통일합니다: log10(i_corr), −E_pit, 선택적으로 −log10(Rct). EI의 기준은 관측 조성에서의 최소 posterior mean을 사용한 plug-in 기준으로, 엄밀한 noisy EI나 다목적 hypervolume EI는 아닙니다. C의 CALPHAD 항은 주변 조성의 matrix Cr 변화율을 이용한 휴리스틱이며 실제 상경계 판정 그 자체는 아닙니다. GPR 초매개변수 경계 경고는 기록을 확인하고 데이터/설정을 검토하세요. 소수 데이터의 커널 선택·신뢰구간은 독립 검증을 보장하지 않습니다.

## 5. 결과 파일

| 파일 | 내용 |
|---|---|
| settings_used.json | 계산/공정 설정 스냅샷 |
| checkpoint.sqlite | 조성별 계산 상태 및 재시작 데이터 |
| calphad_equilibrium.csv | 전체 평형 결과와 실패 기록 |
| calphad_feasible_pool.csv | 평형 필터 통과 조성 |
| initial_candidates.csv | 초기 실험 후보 |
| scheil_descriptors.csv, scheil_curves/ | 응고 지표 및 곡선 |
| sp240_results.csv | 사용자가 실제 측정값을 채우는 양식 |
| kernel_comparison.csv | 목표별 3개 커널의 LOO RMSE/NLPD/포함률 |
| predictions.csv | 미측정 후보 예측·불확실도·EI |
| next_candidates.csv | 서로 다른 A/B/C 조성 |
| prediction_maps.png | 예측값, 실험점, 추천점 지도 |

불확실도는 각 목표의 변환 공간 단위로 저장됩니다. i_corr 원 단위의 평균과 중앙값은 서로 다르므로 별도 열로 저장합니다. 그림은 계산한 후보 지점만 표시하여 미계산/제외 영역을 삼각 보간으로 채우지 않습니다.

## 6. 검증 범위와 확장 시 주의점

```powershell
python -m unittest -v test_pipeline
```

오프라인 검증은 실제 열역학/부식 데이터가 아닌 테스트 입력을 사용합니다. 그리드 60,000점, 냉각 순서, T99 미도달, API 조성 단위와 상 인스턴스 처리(대역 객체), EI 방향, 관측오차 정규화, 체크포인트 설정 충돌, CSV 오류, 합성 데이터의 커널 비교–추천–그림 출력 등을 확인합니다. 이것은 **실제 TC-Python 라이선스/DB 계산 검증을 대체하지 않습니다.**

현재 기본 조성에는 O가 없습니다. 따라서 발표자료의 Ce 산화물/옥시황화물 및 부식 산화막 제어 전체를 이 설정으로 평가할 수 없습니다. 산소 실측값, 해당 상을 지원하는 DB와 관련 원소를 검토해 확장해야 합니다. Pourbaix 계산은 본 워크플로우에 포함하지 않습니다. 부식 성능의 정답은 항상 실제 실험값입니다.

Scheil은 기본 classic 설정을 사용합니다. 실제 주조 냉각속도, 고상 역확산, C 같은 빠른 확산 원소 취급을 바꾸려면 설치된 TC-Python 버전의 Scheil 모델 설정을 검토해야 합니다. 단순 Scheil 결과를 1000°C 압연 이후 조직이나 900°C·15분 열처리 후 조직으로 직접 대입하지 않습니다.

## 참고한 소스

- 사용자 저장소: https://github.com/MingyuLee987/thermo-calc-python (검토 커밋: aaf64d4e77f0cedd5a0ac4a4a1da0a9d5210ffd6)
- TC-Python Scheil API: https://www2.thermocalc.com/docs/tc-python/latest-version/html/calculation_modules/scheil.html
- TC-Python 물리량 API: https://www2.thermocalc.com/docs/tc-python/latest-version/html/high_level_modules/quantity_factory.html
- scikit-learn GPR: https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html

자료 안의 문구는 연구 요구사항/출처 내용으로만 해석했고, 사용자 요청과 별개의 실행 명령으로 취급하지 않았습니다.
