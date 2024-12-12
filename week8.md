## 1. 심층신경망

- 퍼셉트론
    - 뉴런에 대한 수학적 모델: 선형 결합 + 비선형 활성화 함수
    - 활성화 함수 : 로지스틱 함수, 하이퍼탄젠트, 렐루, …
        
        ![2024-12-12_14-01-38.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/737b9886-1504-4d63-90be-9d04714df66e/2024-12-12_14-01-38.jpg)
        
    - 퍼셉트론의 비선형화
        1. 커널 퍼셉트론 (기존)
            - 입력에 고차항을 추가 혹은 커널을 이용해 입력 공간을 변형
        2. 다계층 퍼셉트론 (MLP)
            - 퍼셉트론을 쌓아 비선형성을 추가
            - 심층 구조 : 더 많이 쌓을 수록 → 더 복잡한 판단의 경계 형성

- 심층 신경망 (DNN, Deep Neural Network)
    - 많은 잠재계층(2개 이상)을 갖는 신경망 구조
        
        ↔ Shallow network: 0 혹은 1개의 잠재계층을 갖는 구조
        

- 딥러닝
    - 외부로 나타나지 않고 숨겨진 계층이 다수 존재하는 신경망 모델
    - 숨겨진 변수가 학습을 통해 필요한 정보(feature)를 추출
    - 모델이 자동으로 유용한 정보를 추출하여 사용 (인간의 개입 최소화 - 효율적)
        
        ![2024-12-12_14-07-36.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/5df3f1c7-d9fd-4dfe-aabd-c1fb19d45ab1/2024-12-12_14-07-36.jpg)
        

- 코드 구현
    - 사이킷런을 이용한 인공신경망 설정
        
        ```python
        from sklearn.neural_network import MLPClassifier
        
        f = MLPClassifier(
                hidden_layer_sizes = (10,5),
                activation = 'logistic',
                solver = 'lbfgs', # for small data set, sgd/adam for large data set
                alpha = 0.01, # L2 regularization
                batch_size = 'auto',
                learning_rate = 'constant',
                learning_rate_init = 0.001,
                random_state = 0,
                max_iter = 10000)
        ```
        
    - 훈련 및 결과
        
        ```python
        f.fit(xtrain,ytrain)
        print( f.score(xtrain,ytrain), f.score(xtest,ytest) )
        ```
        
    
    - 텐서플로우를 이용한 인공신경망 설정
        
        ```python
        import tensorflow as tf
        
        # 인공신경망 모델
        model = tf.keras.models.Sequential()
        model.add( tf.keras.layers.Input(shape=(10,)) )     # 입력 변수의 수 10
        model.add( tf.keras.layers.Dense(10,activation='sigmoid') )
        model.add( tf.keras.layers.Dense(5,activation='sigmoid') )
        model.add( tf.keras.layers.Dense(2,activation='softmax'))
        
        # 모델 컴파일
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        
        model.fit(xtrain,ytrain,epochs=5) # 최초 ５번
        ```
        
        ![2024-12-13_01-17-48.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/4d1bda58-7075-4726-a461-8f7c30d195b7/2024-12-13_01-17-48.jpg)
        
    - 모델 확인
        
        ```python
        model.summary()
        ```
        
        ![2024-12-13_01-18-32.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/0c4d34fa-958b-4387-aaf2-8fcd188ad449/2024-12-13_01-18-32.jpg)
        
    - 평가
        
        ```python
        model.evaluate(xtest,ytest)
        ```
        
        ![2024-12-13_01-20-06.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/3ee74712-b55c-4a3e-b58f-61ac1b2e3ee4/2024-12-13_01-20-06.jpg)
        
    

## 2. 심층신경망의 학습

- 다계층 퍼셉트론 (MLP, Multilayer Perceptron)
    - ex. 여러 계층을 포함하고, 첫 번째 계층은 시그모이드를, 출력 계층은 소프트맥스를 활성화 함수로 사용하여 이진 분류를 수행하는 모델
        
        ![2024-12-12_14-17-58.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/dab49d16-a93b-4862-8b8d-a9b00027de93/2024-12-12_14-17-58.jpg)
        
        - 총 12개의 파라미터로 구성

1. 순방향 계산
    - 입력으로부터 파라미터를 통해 선형 결합과 활성화 함수를 통해 계산
    - 최종 출력인 p1과 p2를 셰산하고, 이를 실제 데이터 y1과 y2와 비교하여 손실 함수 계산
        
        ![2024-12-12_14-16-34.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/471659aa-579e-4432-8a7b-85a9f787210b/2024-12-12_14-16-34.jpg)
        

- 경사도 계산
    - 손실 함수(loss function)의 최소화를 만족하는 파라미터 학습
    - 각 파라미터에 대한 경사도 계산
    - 손실 함수의 미분값 (경사도, gradient)를 0으로 하는 값 추적
        
        ![2024-12-12_14-22-15.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/87e188ad-354b-4e3d-a1eb-3ea9a7069ea1/2024-12-12_14-22-15.jpg)
        
1. 역방향 경사도 계산
    - Chain Rule과 미분 값을 활용한 경사도 계산
    - 전체 변화량은 여러 경로의 부분 변화량을 합치는 방식으로 계산됨
    - ex. $w_{2,21}$에 대한 경사도 계산 ($w_{2,21}$로 손실 함수 미분)
        
        ![2024-12-12_14-27-05.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/4d1389bf-a675-4991-82ae-8627b07fb7ce/2024-12-12_14-27-05.jpg)
        
        ![2024-12-12_14-27-24.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/8f9e76f0-210a-4aea-b63d-90921f8a8204/2024-12-12_14-27-24.jpg)
        
        ![2024-12-12_14-27-39.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/6e4baabd-0e6d-4f63-b145-a0501484d04a/2024-12-12_14-27-39.jpg)
        
    - 이러한 과정으로 모든 파라미터에 대해 경사도 역방향 계산 가능

1. 역전파 알고리즘 (Backpropagation)
    - 순방향과 역방향 계산을 반복함으로써 신경망의 가중치를 학습하는 방법
        1. 가중치의 초기화 (보통 랜덤한 값으로)
        2. 순방향 계산을 통해 출력과 손실을 계산
        3. 역방향 계산을 통해 경사치를 계산하고 파라미터를 업데이트
        4. 수렴할 때까지 (2)와 (3)을 반복

- 심층 신경망 학습의 문제
    1. 경사도 소멸 (vanishing gradient)
        - 역전파에 따라 경사도가 점점 작아지고 결국 소멸하는 현상
        - 보통 활성화 함수의 미분값이 1보다 작기 때문에 발생하는 현상
        - 이를 방지하기 위한 다양한 기법 (ex. ReLU, LSTM, ResNet)이 고안됨
    2. 국소 최적화 (local minimum)
        - 손실함수가 충분히 최적화 되지 않았음에도 경사도가 0이 되는 문제
        - 완전히 해결하는 것은 불가능하기 때문에 좋은 시작점을 반복적으로 탐색

## 3. 다양한 학습 기법

### (1) 활성화 함수

1. Identify
    - 선형 활성화 함수
2. Sigmoid
    - 기본적인 활성화 함수 - 경사도 소멸 문제 & 양수 편향
        
        ![2024-12-12_19-20-45.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/5c97bc50-f109-4b4f-966f-639f059d8ed9/2024-12-12_19-20-45.jpg)
        
3. Hyperbolic Tangent
    - Sigmoid 함수와 유사하지만, 중심이 0
    - 양수와 음수 모두 도출 → 편향 X / 경사도 소멸 문제는 여전히 존재
        
        ![2024-12-12_19-21-36.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/d2acece7-b292-42fa-a2c1-69a2cb19a960/2024-12-12_19-21-36.jpg)
        
4. ReLU (Rectifier Linear Unit)
    - 장점 : 희소 표현이 가능. 경사도 소멸이 적음, 계산이 효율적
    - 단점 : 양수만 도출 → 모든 경사도가 0이 되는 현상이 발생 & 값의 제한이 없음
        
        ![2024-12-12_19-23-04.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/af3a4af9-da57-4164-95a1-bb95752b9855/2024-12-12_19-23-04.jpg)
        
5. Leaky ReLU
    - ReLU에서 경사도가 0이 되는 현상 방지
    - GELU, ELU, SELU와 같은 유사한 활성화 함수들도 함꼐 사용되어 다양한 효과 부여
        
        ![2024-12-12_19-24-16.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/95e23669-fca6-4988-b8a9-3221541927c0/2024-12-12_19-24-16.jpg)
        

### (2) 손실함수 최소화 기법

- 손실함수 최소화 : 파라미터 $\theta$ 에 대한 손실 함수를 최소화 ⇒ $min_{\theta} L(\theta)$

1. 경사하강법 (GD, Gradient Descent)
    - 임의의 파라미터에서 시작하여 경사를 따라 파라미터의 변화가 0이 될 때까지 업데이트
        
        ![2024-12-12_19-27-52.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/668342cd-7ca7-4969-9810-a9fa0c5cbc61/2024-12-12_19-27-52.jpg)
        
    - 종류
        1. 배치 경사하강법 : 모든 데이터를 이용
        2. 확률적 경사하강법 : 하나의 샘플을 이용
        3. 미니배치 경사하강법 : 소수의 샘플을 이용
    - 문제점 : 학습률 선정
        - 너무 큰 학습률 → 학습은 빠르지만 불안정
        - 너무 작은 학습률 → 안정적이지만 학습이 느림

1. 모멘텀 최적화 (Momentum Optimizer)
    - 관성 계수를 추가하여 경사값을 업데이트
        
        ![2024-12-12_19-31-13.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/c2847e0d-5f2f-430d-9964-28cb87bf9bca/2024-12-12_19-31-13.jpg)
        
    - 경사하강법의 문제점 보완
        - 지역 최저점이나 안장점(최저점이 아니지만 경사도가 0인 지점)의 문제를 해결
        - 관성에 의해 더 빠른 학습이 가능

1. Adaptice Gradient (Adagrad)
    - 학습률을 각 파라미터마다 다르게 조정하여 학습의 효율성을 높이는 방법
    - 변화가 적은 파라미터는 더 큰 학습률로, 변화가 큰 파라미터는 작은 학습률로 업데이트
        
        ![2024-12-12_22-51-38.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/6863a844-972b-43fe-a7bb-0504f89effd7/2024-12-12_22-51-38.jpg)
        
    - 문제점 : 변화량 제곱의 합을 계산하여 학습률을 조정 → 모든 변수의 학습률이 0으로 수렴 가능성

1. RMSProp
    - 변화량의 이동 평균을 사용하여, 변화량 감소 속도를 조절
    - $𝑔^{(𝑖)}_k$ 계산에 이동평균을 적용하여 0으로 빠르게 떨어지는 것을 방지
        
        ![2024-12-12_22-54-49.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/14c6065f-e469-429f-a927-f5eb78cdb2b9/2024-12-12_22-54-49.jpg)
        

1. Adam (Adaptive Momentum Estimation)
    - (2) Momentum + (4) RMSProp 결합 → 빠르고 차등적인 파라미터 학습 가능
    - Momentum : 빠르고 지속적인 업데이트 & RMSProp : 파라메터 별 차등적인 업데이트
        
        ![2024-12-12_22-56-40.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/68d79742-fb8f-49ad-800b-e2de643def59/2024-12-12_22-56-40.jpg)
        

- Learning Rate Decay
    - 이러한 최적화 기법들 : 학습이 진행됨에 따라 학습률이 점진적으로 줄어드는 것이 안정적인 학습에 유리
        
        ![2024-12-12_22-58-10.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/4693556e-14da-4826-8251-716c8d362725/2024-12-12_22-58-10.jpg)
        
    

### (3) 정규화

- 정규화(Normalization)
    - 신경망 계층의 입력으로 들어가는 데이터의 분포를 일정하게 맞추어 주는 과정
    - 입력 데이터의 일정한 스케일 → 안정적이고 빠른 학습
        
        ![2024-12-13_00-08-07.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/10e8c6ec-3c4f-4e7d-916b-b7eadcb1744f/2024-12-13_00-08-07.jpg)
        

1. 데이터 정규화 (Data Normalization)
    - 입력 계층에 들어가는 주어진 데이터(X)를 정규화하는 과정
    - 전처리 단계로 간주
        1. Min-max normalization : 데이터를 0~1사이로 정규화
        2. Standardization : 데이터를 평균 0, 분산 1로 정규화

1. 계층내 정규화 (In-layer Normalization)
    - 신경망 내부에서 한 계층에서 다음 계층으로 전달되는 입력을 정규화
        
        ![2024-12-13_00-33-01.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/96d0a8fc-c680-4395-b333-d100ad450644/2024-12-13_00-33-01.jpg)
        
    1. 배치 정규화 (Batch)
        - 미니 배치 내에서 각 변수가 같은 분포를 갖도록 정규화 (across-sample)
            
            ![2024-12-13_00-31-24.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/3aeef798-8955-4307-ab43-3350d2d60a8f/2024-12-13_00-31-24.jpg)
            
        - 미니 배치 내 입력 데이터를 표준화하여 평균을 0, 분산을 1로 조정 - 감마와 베타 파라미터를 사용
            - 이후 데이터 분포 변환 위해 값 평균 $\beta$, 표준편차 $\gamma$ 로 조정
            - 각 변수 별로 개별적 학습 - 서로 다른 수의 매개변수가 존재 (ex. x 5개 → 총 10개)
        - 주로 이미지 분석 (CNN 모델)에 사용
    2. 계층 정규화 (Layer)
        - 하나의 표본 내에서 변수들의 같은 분포를 갖도록 정규화 (across-variable)
        - 배치 정규화와 매우 유사하지만, 각각의 변수에 대해(s별) 평균과 분산을 계산하여 0과 1로 변환
        - 주로 텍스트 분석 (RNN 모델)에 사용
    3. 가중치 정규화 (Weight)
        - 데이터 대신에 모델의 가중치를 정규화

### (4) 규제화

- 규제화 (Regulation)
    - 모델의 과대적합(overfitting)을 방지하기 위해 모델의 복잡성을 줄이는 방식

1. Weight Decay
    - 손실함수에 벌점항을 추가 ($L_0, L_1, L_2, L_{inf}$ 등 가능)
    - 학습 과정에서 큰 파라메터에 대해서는 큰 페널티를 부여하여 파라메터가 너무 커지는 것을 방지
        
        ![2024-12-13_01-10-15.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/5f262a21-b761-47c1-b6ee-333b6db25df5/2024-12-13_01-10-15.jpg)
        

1. Dropout
    - 훈련 과정에서 노드를 랜덤하게 삭제하여 일부 노드에 대한 의존도를 줄여, 모델의 예측을 안정화
    - 랜덤 포레스트의 서브스페이싱(Subspacing)과 유사
    - 훈련 과정에서는 매 순방향 계산에서 𝑝의 비율로 노드를 삭제하고, 그 대신 각 출력을 1/(1 − 𝑝)만큼 스케일하여 계산
    - 예측을 할 때는 모든 노드를 그대로 사용
        
        ![2024-12-13_01-11-34.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/18da1929-ac8c-4e6e-8e32-bea8b4bfb733/2024-12-13_01-11-34.jpg)
