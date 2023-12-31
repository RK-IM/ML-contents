입력 문장에서 분위기나 특징을 찾아내 그에 맞는 식당을 추천해주는 코드입니다.  

## 전체   
<img src="https://github.com/RK-IM/ML-contents/blob/main/Projects/restaurant_recommendation/images/pipeline.png">


## 출력 결과  
<img src="https://github.com/RK-IM/ML-contents/blob/main/Projects/restaurant_recommendation/images/sample_result.png" height="400">


결과 실행은 sbert_module 안에 있는 evaluate.py 파일을 통해 할 수 있습니다. 
예측을 위한 입력은 --text 인자를 통해 넘겨주면 됩니다. 
```zsh
python evaluate.py --text "예시 문장"
```

※ 현재 파일에는 깃헙 용량 제한에 의해 식당 리뷰 파일과 학습에 사용된 모델이 추가되어있지 않습니다. sbert_trainer.py와 tariner.py를 실행해서 사전 학습 모델을 이용해 학습할 수 있습니다. 학습이 완료된 모델은 현재 시각으로 저장되므로 config 파일을 수정해 예측에 사용되는 모델을 지정해주면 됩니다. 학습에 필요한 식당 리뷰 파일은 진행한 프로젝트 서버에서 받아와야 합니다.  

## 파일 설명

- review_tfidf.ipynb: 리뷰 데이터에서 라벨로 사용할 키워드를 tfidf로 찾아보는 과정입니다. 여기서 선정한 키워드와 다른 실제 사용중인 서비스, 직접 리뷰를 읽어보면서 최종 키워드를 선정했습니다.
- text_DataAugmentation.ipynb: 리뷰 데이터를 증강하는 과정입니다. Bert 모델을 활용한 방법과 Word2Vec와 직접 추가한 단어를 이용해 증강하는 방법 2가지를 사용했습니다. 증강은 Random swap과 Synonym replacement 두 가지를 사용했습니다.
- compare_text_eda.ipynb: 텍스트 데이터 증강 결과를 확인하기 위한 과정입니다. 학습에는 라벨링 된 800개 정도의 데이터가 사용되었습니다. 증강은 이 데이터로 600개 정도의 문장을 만들어 직접 라벨링 한 리뷰만으로 학습한 모델과 비교했습니다. 모델 검증에는 학습에 사용되지 않은 1800 여 개의 리뷰를 사용했습니다.
- Evaluate_all.ipynb: 학습한 모델로 전체 리뷰에 대한 키워드들의 점수 변환 과정입니다.
- sotre_score.ipynb: 식당에 키워드별 점수와 키워드를 부여하는 과정입니다. 단순히 리뷰에서 나오는 키워드 확률을 평균해 식당에 부여한다면, 낮은 비율로 존재하는 키워드가 과소평가 될 위험이 있습니다. 때문에 중간 과정을 거쳐 점수를 부여했습니다(클래스 가중치 곱, 스케일 조정).

---

- model_results.csv: 프로젝트 진행 중 사용한 모델들의 점수 결과입니다. [텐서보드](https://tensorboard.dev/experiment/YV5Jg0MAT6OVMbBPPMSEjw/#)에서 그래프로 볼 수 있습니다.

---

### sbert_module

- config.py: 프로그램에 사용된 하이퍼파라미터들과 파일 이름, 경로 등입니다.
- data_setup.py: 학습에 사용하기 위한 데이터를 가져오기 위한 코드들이 작성되어있습니다. 데이터 불러오기, 전처리, 훈련.테스트 세트로 나누기, sbert 학습을 위해 데이터 프레임 형태 변환, 데이터셋과 데이터 로더 정의가 있습니다.
- engine.py: 파이토치 학습을 위한 모듈입니다. 훈련 과정, 테스트 과정, 두 과정을 합친 함수가 있습니다. 학습을 진행하면서 출력 결과를 model 파일에 저장하게 됩니다.
- evaluate.py: --text로 문장을 입력받았을 때 각 키워드마다 확률을 출력해줍니다. 데이터베이스에 있는 식당들의 점수와 비교하여 가장 유사한 식당 5개도 식당의 점수와 함께 출력해줍니다. 유사한 식당은 코사인 유사도로 측정합니다.
- massive_evaluate.py: 학습한 모델로 모든 리뷰를 예측하는 파일입니다. 결과는 config 파일에 명시된 파일 경로와 이름으로 저장됩니다.
- model_builder.py: 학습에 사용한 모델입니다. 여기서 sbert 구조와 분류기의 구조를 변경하면 됩니다.
- sbert_trainer.py: sbert를 훈련시키는 파일입니다. 학습한 모델은 학습이 시작된 시간의 이름으로 생성된 폴더에 저장됩니다.
- trainer.py: 앞에서 학습한 sbert 모델을 가져와 분류기를 학습하는 과정입니다. 여기서 각 클래스마다 점수들이 측정됩니다. 측정되는 점수는 Loss, F1 score, ROC AUC, PRAUC 입니다.
- update.py: 새로운 데이터가 추가되었을 때 모델을 업데이트 하기 위한 파일입니다. 업데이트 시 --update를 같이 넘겨준다면 사용하는 모델이 현재 학습한 모델로 변경됩니다. config 파일이 수정되며 이전에 사용한 모델은 삭제됩니다 (용량 문제).
- utils.py: 훈련의 편의를 위한 파일입니다. 성능 측정을 위한 점수 계산과 모델 저장을 위한 함수들이 작성되어있습니다.

---

- model: 분류기 모델과 측정 결과들이 저장되어있습니다.
- sbert_model: 학습한 sbert 
