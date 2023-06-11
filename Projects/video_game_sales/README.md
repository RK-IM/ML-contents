## 개요

2016년까지의 게임 판매량 데이터를 분석한 프로젝트 입니다.  
이를 통해 추후 출시할 게임의 장르를 선정합니다.

## 데이터

게임 판매량은 1970년대부터 2016년도까지 북미, 유럽, 일본, 그 이외 지역으로 나뉘어 기록되이었습니다. 분석에는 전체 데이터 중, 마지막 연도인 16년도에 사용중인 플랫폼 중 가장 출시년도가 빠른 플랫폼의 출시일부터 사용했습니다(플랫폼은 X360, 2005년 부터 2016년 까지).

## 파일 설명

- vgames2_clean: 캐글에서 받아온 전체 게임 데이터셋 중 전처리가 완료된 csv 파일입니다. 전처리 과정은 같은 폴더에 있는  [preprocessing.ipynb](https://github.com/RK-IM/Projects/blob/main/video_game_sales/preprocessing.ipynb)에 있습니다.
- all_games.csv: 분석에 사용한 외부 데이터 입니다. 메타크리틱이라는 평가 사이트에서 진행하는 게임 평가 점수 데이터로, 사용자와 평론가들의 점수로 추가적인 분석을 위해 가져왔습니다. 

- preprocessing.ipynb: 캐글 원본 데이터 전처리 과정입니다.
- vgames_EDA.ipynb: 전처리가 완료된 데이터로 분석을 진행한 파일입니다.

## 분석

1. 지역별 선호 장르
2. 연도별 선호 장르
3. 출고량 변화
4. 평가 점수별 출고량  

에 대한 분석을 진행했습니다. 구체적인 과정은 [vgames_EDA.ipynb](https://github.com/RK-IM/Projects/blob/main/video_game_sales/vgames_EDA.ipynb)에 있습니다.

- 지역별 선호 장르 비교  
<img src="https://user-images.githubusercontent.com/94027045/220020075-088a61a4-2fc4-4f4b-969c-bf1e729c691f.png" width=600>

- 연도별 출고, 판매량 비교  
<img src="https://user-images.githubusercontent.com/94027045/220020213-3eee7829-bf7b-4859-b8de-d1a6451edab6.png" width=600>  

- 메타 점수와 판매량 사이 비교  
<img src="https://user-images.githubusercontent.com/94027045/220020731-e82962a8-af03-4a57-b358-fd56d978b302.png" width=600>  
