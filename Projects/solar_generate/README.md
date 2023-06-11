## 개요
2017년부터 2021년까지 일사량을 기반으로 태양광 패널 설치시 절약할 수 있는 전기세를 예측하는 프로젝트입니다.

## 데이터
기상청 기상자료개방포털에서 2017년부터 2021년까지 부산지역의 일사량 데이터를 사용했습니다. 발전량은 이 일사량과 패널의 넓이, 효율을 고려해 계산했습니다. 전기 요금 산정 방법은 2022년 10월 기준입니다.

## 파일 설명

- dataset.txt: 2017년부터 2021년까지 부산지역의 일별 날씨 정보입니다.
- webpage.txt: 배포 페이지 url 입니다. 현재는 헤로쿠 무료 사용기간 종료로 보이지 않습니다.
- solar.csv: 위 원본 데이터에서 날짜, 일조량, 일사량, 기압만 가져온 csv 파일입니다.
- usage.csv: 사용한 전기 요금별 비용과 발전기 사용으로 절약된 요금 입니다. 대시보드 출력용으로 작성되었습니다. 11월 기준, 발전량은 약 275kW 입니다.
---
- collect_and_upload_db.ipynb: 데이터베이스에 데이터들을 적재하는 과정입니다. 공공데이터 api를 사용했으며 몽고 DB에 raw data를 적재했습니다. 적재된 raw 파일을 가져와 필요한 데이터만 PostgreSQL로 가져오고 대시보드 출력을 위한 간단한 계산 과정을 진행합니다. 헤로쿠로 메타베이스 배포를 위해 엘리펀트SQL에도 추가해주었습니다. 
- EDA_ML_Calc.ipynb: PostgreSQL에서 데이터를 가져오고 일사량 예측과 발전량 계산, 예상 요금을 계산하는 과정입니다.
- model.pkl: 배포될 모델 피클 파일입니다
---
- docker-compose.yaml: 도커에서 연 PostgreSQL과 메타베이스를 연결해주기 위한 yaml 파일입니다.

## 결과 예시

- 홈 화면
<img src="https://user-images.githubusercontent.com/94027045/220024837-e98cc86f-3802-4be8-aaab-acd89cc70719.png" width=600>

- 예측 화면
<img src="https://user-images.githubusercontent.com/94027045/220030823-2736fabf-ef7f-4a63-9a7d-cc34d2837dbb.png" width=600>

<img src="https://user-images.githubusercontent.com/94027045/220030919-24beb265-993d-4e2e-af06-0ffe49ff78a6.png" width=600>
