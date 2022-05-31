# Task 설명
## [Dacon] 한국어 문장 관계 분류 경진대회

※ 참고 : 모든 코드는 Colab Pro 환경에서 수행되었습니다.

# Result
Public  : 0.905
Private : 0.89915
Awards  : 2위

# Train

```
```


# Inference
```
```


# 시도했던 전략들

## 1. Bert 혹은 PLM을 현재 Data의 Input Sequence들에 추가 training 한 뒤에 Down Stream Task 진행해보기

## 2. Back Translation

- PaPago가 1순위
- Pororo 2순위

> 참고 : https://dacon.io/competitions/official/235747/codeshare/3054?page=1&dtype=recent
- 번역 API 입장에서 처음 보는 단어(OOV) 특수문자 혹은 문장의 어투 등 유지하기 위해선 PaPago가 가장 좋음
    - 기존 문장의 의미를 깨트리지 않는 성능 자체도 제일 좋은 듯..
- 하지만 긴 문장을 너무 짧게 번역 생성해버린다던가 하는 문제점도 존재
- 따라서, PaPago로 대부분을 처리하고, Pororo는 PaPago로 좋은 번역문장이 안 나왔을 때 대체품으로 사용.
- 이렇게 데이터 증강을 할 때 중요한 점은 Valid Eval시 Leakage가 존재하지 않게끔 train val split을 잘 해야함.


## 3. 모델 여러가지 비교해보기

> 리더보드 참고 :
- https://klue-benchmark.com/tasks/68/leaderboard/task
- https://github.com/KLUE-benchmark/KLUE#baseline-scores
- 아무래도 klue/roberta-large 미만 잡인듯.

- BERT
- ELECTRA
- ROBERTA
- XLM - ROBERTA (small, base, large)
- klue/roberta-base (small, base, large)

※ 혹시 Multi Task를 학습한 모델이 언어 자체에 대한 이해를 더 잘 할수 있지 않을까?
- MT-DNN
- mT5
- ke-T5

※ nli Task에 Fine-Tuning 되어있는 모델을 사용해보자.
- roberta-large-mnli
- joeddav/xlm-roberta-large-xnli
- Huffon/klue-roberta-base-nli (이미 Klue 데이터에 Fine-Tuning 되어있어서 Valid 계산시 Data Leakage 현상 존재)

※ nli Task는 아닌데, Classification Task에 Fine - Tuning된 모델
- bespin-global/klue-roberta-small-3i4k-intent-classification (좋지 않은 성능)



## 4. K-Fold Ensemble 진행해보기
- Fold를 더 늘리면 일반화되고 score 잘될 것은 분명하지만, 시간이 너무 오래 걸릴 것.
- 현재 Fold=5로 수행하는데도 너무 오래걸림
- 일단 Fold==5로 계속 실험들을 이어나가도록 하고,
- 대회 후반에 순위권 점수차가 너무 미미해서 조금이라도 성능 올리고 싶으면 Fold 수를 높이는 걸 시도해보자

## 5. [CLS] Token 뿐만 아니라 [SEP] Token도 같이 활용하면 좋지 않을까?
- Relation Extraction Task할 때에는 NER (개체명인식) 활용해서 Input의 주요 키워드 양 옆에 Special Token 추가해봤었는데
- NLI Task에는 Input의 주요 키워드라는게 없지..

## 6. 마지막 Layer에 LSTM (아니면 GRU) 추가해보기
- 시간리소스가 대폭 증가할 것이란 걸 감수..

## 7. Input에 Noise를 추가해보기
> 참고(BART 논문) : https://arxiv.org/pdf/1910.13461v1.pdf

- 문장의 일부 잘라서 앞에 붙이기 (Sentence Permutation)
- Premise 내에서 순서 바꾸기 (Document Rotation)
- Hypothesis 내에서 순서 바꾸기 (Document Rotation)
- 둘을 연결한 concat 내에서 순서 바꾸기 (=> 너무 심한 Noising 방법일듯)
- 문장의 일부 Masking 하기 (Random Masking, + Overlap Masking)
- 혹은 그냥 삭제해버리는 Token Deletion

## 8. Premise와 Hypothesis의 순서를 바꾸어서 Data Augmentation 해보기
- 일단 Entailment와 Contradiction은 불가능할 것 같음.
- 혹시 Neutral 은 가능하..려나..?
- 이렇게 데이터 증강을 할 때 중요한 점은 Valid Eval시 Leakage가 존재하지 않게끔 train val split을 잘 해야함.

## 9. 외부 공공 데이터 추가해보기
- 모두의 말뭉치 KoNLI
- KLUE Official Data 중 dev file 가져와서 사용해보기

## 10. Loss Function 바꿔보기
- cross entrophy
- Focal Loss
- Label Smoothing Factor 증가시켜보기

## 11. HyperParameter 변경해보기
- learning rate (진짜 제일 중요함, 특히 모델 종류에 따라)
- batch size
- scheduler
    - warm up steps
    - gamma
    - cycle_ratio
- optimizer
- dropout

## 12. R-Model Architecture 구현해보기
- R-BERT
- R-ELECTRA
- R-ROBERTA

## 13. Visualization (Wandb Logging)

> Valid에서 예측 틀린 문장들 집중적으로 살펴보기

- https://docs.wandb.ai/guides/data-vis#text
- https://wandb.ai/stacey/nlg/reports/Tables-Tutorial-Visualize-Text-Data-Predictions---Vmlldzo1NzcwNzY


## 14. 문장 유사도 측정하고 활용
- 문장 pair의 유사도가 높으면서, Label이 Contradiction 혹은 Neutral 일때 모델이 많이 헷갈리지 않을까?
- 일단 이를 확인해보고 만약 그렇다면, 이런 문장 pair 샘플들에 모델들이 더 집중할 수 있는 학습방법을 고려해봐야 할 것이다.
- 일정 유사도를 넘으면서, Contradiction인 데이터들
- 유사도 측정 기준 : jaccad , cosine similarity 혹은 SST 모델 활용 등
> link : https://dacon.io/competitions/official/235747/codeshare/3054?page=1&dtype=recent

- 혹시 그냥 해본 생각인데, 유사도를 측정한 뒤 0.9라는 값이 나오면 (이 때는 높다고 판단)
- [CLS] 문장1 [SEP] 문장2 [SEP] 두 문장의 유사도는 높다 <= 이런식으로 suffix를 추가해보는 것은 어떨까?
- [CLS] 문장1 [SEP] 문장2 [SEP] 두 문장의 유사도는 0.9 <= 혹은 이런식으로 숫자 값을 input text에 추가해보기

## 15. Epoch 늘리기
- 아직 학습이 덜 된 상태로 종료가 된 듯하다.
    - klue/roberta-large와 같은 모델들 epoch 15~20까지 돌려보기
    - Early Stopping 추가

## 16. 마지막에 추가 학습시키기 (valid 없이)
- 이때까지 시도해서 성능향상 성공한 적은 없었음, 시간 남을때 도전해볼 만한 최후의 보루 느낌?
    - 현재 모델들은 K-Fold 를 활용하며 일부 (80%) 데이터만 가지고 학습된 상태이다.
    - 따라서 모든 train 사용하여 추가 재학습을 하면 더욱 많은 정보를 학습할 수 있지 않을까?
    - 과적합될 수도 있으니, 1이나 2 epoch 정도만이라던가, 기존의 weight들이 무너지지 않는 선에서 학습시켜보기
    - learning rate를 완전히 극도로 낮춰서 시도해보면 좋은 방법일 수 있지 않을까?

## 17. Data Augmentation (jaccad & STS Model 둘다 이용)
- entailment 와 contradiction Label은 이미 premise에 충분한 정보를 가지고 있어서 결정된 데이터임.
- 그렇다면, hypotheis는 건드리지 않고 premise에 텍스트를 추가해주면 라벨이 그대로인 데이터가 생성된다.
- hypothesis와 내용상 무관한 텍스트를 추가해주는 것이 중요할 것이다.
    - 이는 STS 모델을 통해 내용상 무관한 텍스트를 분별해낼 수 있을 것이다.
- hypothesis와 최대한 jaccad 유사도가 높은 text 를 집어넣어서 모델이 학습 어려워하게끔? 데이터를 만들어주는 게 유효하지 않을까?
    - 이건 일단 실험을 해보아야 알듯, 쉽게 만들어줘야 모델이 더 좋아할 수도.
    - 만약 이렇게 만들고 싶다면 (모델이 어려워 하게끔), 자카드 유사도 측정 후 가장 유사한 text를 추가할 수 있을 듯.
- 이렇게 데이터 증강을 할 때 중요한 점은 Valid Eval시 Leakage가 존재하지 않게끔 train val split을 잘 해야함.
    - 아래 18번 사항 참고

## ※ DataAugmentation 적용 후, 기존 데이터셋을 대체할 것인지 혹은 그냥 추가로 냅다 넣을 것인지 검증 필요.

## 18. Valid Score 측정시, Data Leakage 문제
- 혹시 Valid Data에서는 점수가 높게 나오는데 리더보드에서는 점수가 낮게 나타나는, 차이가 나는 이유는 무엇일까?
    - 현재 적게는 0.03, 많게는 0.05 정도까지 차이남.
- 그렇다면 Validation 측정시 Data Leakage가 있다는 가정을 해볼 수 있지 않을까?
- 실제로 Train Data를 보면 너무 비슷한 문장들을 돌려막기로 데이터를 추가한 감이 쪼끔 있음. 
    - (물론 Train 측면에서 어떤 좋은 점도 있겠지만.. Valid Split시 문제가 있지않나?)
    - 예를 들면 동사와 단어, 혹은 부사만 바꾼 다음 라벨링을 새로 하는 식으로 일부 Train Data가 구성되어 있는 감이 있음.
    - 만약 이러한 부분을 고려하지 않고 그냥 무작위로 train valid split을 하게 되면
    - train에서 봤던 premise와 매우 유사한 premise가 valid에 있을 때 너무 쉽게 맞추게 된다. (hypothesis가 조금 달라서 라벨이 서로 다른 데이터라 하더라도..)
    - (네이버 부캠 Pstage1 - Mask Image Classification에서 Validation시 Data Leakage가 발생하는 것을 상기해보자.)
- 그렇다면 이들을 그룹화하고 내가 직접 합리적으로 train - valid 나누는 기준을 설정하면 valid score 측정을 좀 더 일반화할 수 있지 않을까?
    - 그룹화의 예시로는 Premise를 기준으로 아예 같은 Text끼리 묶거나 Jaccad 유사도만 써도 가능할 듯.

- 기존 Train Set 말고 추가된 KLUE Official Dev Set도 Valid를 구축하고 싶다면, 아래를 참고
    - KLUE Official Dev Set의 Premise 기준 그룹을 묶으면 nsize=3임.
    - 그럼 다음과 같이 임의로 train valid를 나눠볼 수 있을 듯.
    - 1) 각 그룹(3개) 중 2개는 train & 1개는 valid 에 넣는 방식
    - 2) 총 1000개의 그룹 중 그룹별로 train - valid split하기

## 19. Max Seq Length 줄이고, Batch Size 늘리기
- Input의 max seq len을 줄이면, 메모리 여유생길 것.
- 현재 128인데 -> 96 정도로는 줄여도 되지 않을까?

## 20. 특수문자 제거하기
- Test Set에는 없는데 Train Set에만 있는 특수문자 포함 데이터들이 있음.
- 이런 데이터는 학습을 방해하지 않을까?


## ※ 이외 시도해보지 못한 아이디어들

> Test Set Pseudo-Labeling 활용해보기
- 이번 대회 규정에는 허용되지 않는 방법


## ToDo List
- [ ] Split train 1-epoch code
- [ ] Train Single-fold code
- [ ] Add Inference code (Single / K-Fold Sperately)
- [ ] Cleaning wandb logging code
- [ ] Add Idea Description Picture
- [ ] Apply Data Augmentation to Train code directly