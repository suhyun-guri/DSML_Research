# DSML_Research
**DS&ML (Data Science&Machine Learning) 센터 소속 학부 연구생 프로젝트**
> 프로젝트명 : 의료데이터 예측 모델 개발 + 폐렴환자의 생존에 영향을 미치는 의료 바이오 마커 탐지 (Detection of medical biomarkers affecting mortality in patients with intensive care unit EMR pneumonia)
* 2021.06 ~ 
* 국내/해외 논문을 목표로 함.
* MIMIC_III 데이터를 사용하여 mortality prediction 모델을 구축
* Feature Importance 계산 방식을 개발하여 생존/사망에 영향을 미치는 의료 바이오 마커 탐지

## 📑 Paper.
- [중환자실 폐렴 환자에 대한 시뮬레이션 기반 시계열 사망 마커 탐지](https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE11113850)
  - 김수현, 이수현, 고가연, 안홍렬*
  - 한국정보과학회 2022 한국컴퓨터종합학술대회

## 🌞 MIMIC_III DATA
[MIMIC-III documentation](https://mimic.mit.edu/docs/iii/) <br>
MIMIC-III (Medical Information Mart for Intensive Care III) is a large, freely-available database comprising deidentified health-related data associated with over forty thousand patients who stayed in critical care units of the Beth Israel Deaconess Medical Center between 2001 and 2012.

The database includes information such as demographics, vital sign measurements made at the bedside (~1 data point per hour), laboratory test results, procedures, medications, caregiver notes, imaging reports, and mortality (both in and out of hospital).

## 📊 사용 Table

- PATIENTS (SUBJECT_ID, EXPIRE_FLAG)
- ADMISSION (SUBJECT_ID, DISCHTIME)
- D_ICD_DIAGNOSES (SHORT_TITLE, ICD9_CODE) - 폐렴 병명 코드 추출에 사용
- D_ICD_DIAGNOSES (SUBJECT_ID, ICD9_CODE) - 폐렴 환자 추출
- LABEVENTS (SUBJECT_ID, ITEMID, CHARTTIME, FLAG)
- [생성] 폐렴환자lab.csv (7799, 690), 폐렴환자.csv (7807, 8)
- PRESCRIPTIONS( ) - Feature 추가에 사용
- PROCEDUREEVENTS_MV( ) - Feature 추가에 사용

## What did I do.
- 학부 연구생 리더로 전체적인 진행을 주도
- MIMIC-III 의료 빅데이터 분석 및 가공 (환자, Timepoint, Items로 이루어진 3차원 데이터)
- 여러 모델 적용 연구 및 최적의 성능을 보이는 모델(LSTM) 선정
- LSTM 모델을 활용한 폐렴 환자 생존 예측
- 의료 마커 탐지를 위해 Feature Importance 계산 방식 개발 및 적용

## Tech Stack.

Python, Pandas, Numpy, Scikit-Learn, Tensorflow, Keras
