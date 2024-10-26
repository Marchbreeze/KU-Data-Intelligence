# [ SVM ]

- SVM (Support Vector Machine, 지지벡터머신)
    - 분류 : SVC (Support Vector Classifier)
    - 회귀 : SVR (Support Vector Regression)

- 주된 성질
    1. 최대 마진 분류기 (Maximal margin classifier)
    2. 소프트 마진 (Soft margin)
    3. 커널 (Kernel)

## 1. 최대 마진 분류기

### (1) 선형 판단 경계

- 이진 분류 문제의 경우
    
    ![2024-10-25_17-08-35.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/a421b555-95c6-495c-8b89-7cdd98f8a343/2024-10-25_17-08-35.jpg)
    
    - 두 클래스 또는 다수의 클래스를 **선형 방정식**을 통해 나누는 경계선
    - $\beta_0 + \beta_1x_1 + \beta_2x_2 + \dots + \beta_nx_n = 0$           ($\beta_1^2 + \beta_2^2 + \dots + \beta_p^2 = 1$)
    
- 두 분류를 완변히 분리하는 경계의 조건
    - 한쪽 분류를 𝑦 = 1로 다른 한쪽 분류를 𝑦 = −1로 표시하면,
        
        $y_i(\beta_0 + \beta_1x_{i1} + \beta_2x_{i2} + \dots + \beta_px_{ip}) > 0$
        
    - 선형 경계는 많이 찾을 수 있음 → 최적의 경계 선택 필요
        
        ![2024-10-25_17-12-58.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/bab69eea-c15d-4df4-9968-1b45d7a926b1/2024-10-25_17-12-58.jpg)
        

### (2) 최대 마진

- `최대 마진 분류기`
    - SVM은 가장 가까운 점으로부터의 거리가 최대가 되는 선형 판단 경계를 선택
        - $d = y_i(\beta_0 + \beta_1x_{i1} + \beta_2x_{i2} + \dots + \beta_px_{ip})$
    
- `지지 샘플` (Support)
    - 모델을 결정하는 샘플 (가장 가까이 있는 샘플)
    - 각 샘플은 벡터의 형태로 나타나기 때문에 지지벡터로 불림
    - SVM은 모든 샘플을 다 고려하는 것이 아니라 오직 `몇몇 샘플만 고려` = 차별점!
        
        ![2024-10-25_17-17-04.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/82c1822a-241c-422c-8368-3d2856886c88/2024-10-25_17-17-04.jpg)
        

- 수학적 표현
    1. 마진 최소화 표현
        - $\text{max}_{\beta_0, \beta_1, \dots, \beta_p} M \quad \text{s.t.} \quad y_i \left( \beta_0 + \beta_1 x_{i1} + \beta_2 x_{i2} + \dots + \beta_p x_{ip} \right) \geq M$
            - M : 최소 거리를 나타내는 마진
            - 모든 $x_i$ 에 대해, 해당 클래스의 결정 경계에서 마진 이상의 거리를 유지해야 함
    2. 가중치 최소화 표현
        - $\text{min}_{w_1, \dots, w_p, b} \sum_{j=1}^{p} w_j^2 \quad \text{s.t.} \quad y_i \left( b + w_1 x_{i1} + w_2 x_{i2} + \dots + w_p x_{ip} \right) \geq 1$
            - 가중치 벡터 $w$의 크기를 최소화
            - $w_j^2$ : 각 가중치의 제곱 → 마진과 반비례 관계
            - 모든 $x_i$ 에 대해, 경계에서 최소 1의 거리를 유지해야 한
    3. 벡터 표현
        - $\text{min}_{w,b} ||w||^2 \quad \text{s.t.} \quad y_i \left( w^T x_i - b \right) \geq 1$
            - $||w||^2$ : 가중치 벡터 w 의 L2 노름 =  $\sum w_j^2$

## 2. 소프트 마진

- 현실적으로 완벽한 선형 경계를 찾는 것은 불가능함

- `소프트 마진` (Soft margin)
    - 샘플들이 하드 마진(hard margin)보다 안 쪽으로 들어오는 것을 허용
    - 몇몇 예외를 허용함으로써 대부분의 샘플을 올바르게 분류하고 모델의 안정성을 높임
        
        ![2024-10-25_17-18-46.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/ed5f9f90-4240-47c1-ade4-7034f1b62f15/2024-10-25_17-18-46.jpg)
        

- $𝜉_𝑖$ (ksi) : 샘플 𝑖 의 위반 정도 → 전체 위반의 정도를 규제
- $V$ : 튜닝 파라메터 → 전체 위반의 정도를 조절

- 수학적 표현
    1. 마진
        - $\text{max}_{\beta, \xi} \, M \quad \text{s.t.} \quad y_i \left( \beta_0 + \beta_1 x_{i1} + \beta_2 x_{i2} + \dots + \beta_p x_{ip} \right) \geq M(1 - \xi_i)$
        - $\sum_{i=0}^{n} \xi_i \leq V , \quad  \xi_i \geq 0$
    2. 벡터
        - $\text{min}_{w,b,\xi} ||w||^2 \quad \text{s.t.} \quad y_i (w^T x_i - b) \geq 1 - \xi_i$
        - $\sum_{i=0}^{n} \xi_i \leq V, \quad \xi_i \geq 0$
    3. 힌지
        - $\text{min}_{w,b} \quad L(Hinge) + L(L2)$
            - $L(\text{Hinge}) = \text{max}(0, 1 - y_i (w^T x_i - b))$
            - $L(\text{L2})= \lambda ||w||^2$
        
        - `𝜆` : 모델의 복잡성을 결정하는 튜닝 파라메터
            - 𝜆 ↑ (C↓, 𝑉 ↑) : 넓은 마진, 간단한 모델, Underfit의 가능성 높음
            - 𝜆 ↓ (C↑, 𝑉 ↓) : 좁은 마진, 복잡한 모델, Overfit의 가능성 높음

## 3. 커널

- `커널` (Kernel)
    - `비선형적`인 판단의 경계를 도입
    - 선형 모델을 비선형 데이터에 적용하기 위한 기법
    - 고차원의 변수를 생성해 모델에 포함
        
        ex. $𝛽_0 + 𝛽_1𝑋_1 + 𝛽_2𝑋_2 + 𝛽_3𝑋_1^2 + 𝛽 _4𝑋_2^2 + 𝛽_5𝑋_1𝑋_2 = 0$ (2차원 → 5차원)
        

- 커널 변환
    - 커널을 통하여 샘플 공간의 차원을 다른 공간으로 변환
    - 고차원에서 선형 모델을 찾은 후 다시 저차원으로 내리면서 비선형 모델을 구축
        
        ![2024-10-25_17-59-23.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/edbc3cb4-c06c-4349-9305-fe04976fd3fc/2024-10-25_17-59-23.jpg)
        

- 수학적 표현
    - $\text{max}_{\alpha} \left( \sum_{i=1}^{n} \alpha_i - \frac{1}{2} \sum_{i,j=1}^{n} \alpha_i \alpha_j y_i y_j x_i^T x_j \right)$
    - 커널 함수
        - $k(x_i,x_j) = x_i^T x_j$  로 변환하여 다양한 커널 적용 가능

1. 선형 커널
    - 판단경계가 선형인 모델
    - $k(x_i,x_j) = x_i^T x_j$
2. 다항 커널
    - 고차항을 포함하고, 𝑑 가 튜닝 파라미터
    - $k(x_i,x_j) = (x_i^T x_j)^d$
    - 코드
        
        ```python
        from sklearn.svm import SVC
        
        # 정규화 파라미터 3, 다항 커널 이용 & 다항식 차수 3
        f = SVC(C=5, kernel='poly', degree=3)
        f.fit(xtrain,ytrain)
        ```
        
3. Radial basis function (RBF) 커널
    - 무한한 차원으로 확장, 𝛾 가 튜닝 파라미터
    - $k(x_i,x_j) = exp[-𝛾||x_i - x_j||^2]$
    - 코드
        
        ```python
        from sklearn.svm import SVC
        
        f = SVC(C=5, kernel='rbf', gamma='auto')
        f.fit(xtrain,ytrain)
        ```
        
    - c 값을 조절 → 교차검증 그래프를 기준으로 선정
        
        ```python
        # 파라미터 범위 (10^-5 ~ 10^5)
        params = 10**np.linspace(-5,5,num=31)
        
        acc_train = []; acc_cv = []; acc_test = []
        for c in params:
            f = SVC(C=c,kernel='rbf',gamma='auto',random_state=0)
            f.fit(xtrain,ytrain)
            acc_train.append( f.score(xtrain,ytrain) )
            acc_cv.append( cross_val_score(f,xtrain,ytrain,cv=5).mean() )
            acc_test.append( f.score(xtest,ytest) )
            
        idx = np.log10(params)
        plt.plot(idx,acc_train,idx,acc_cv,idx,acc_test)
        plt.legend(['Train','CV','Test'])
        ```
        
        ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/3cf7a7ac-2c44-4ab3-a109-a1edb15c7087/image.png)
        
    - GridSearchCV 사용으로 최적 모델 찾기
        
        ```python
        from sklearn.model_selection import GridSearchCV
        
        params = {'C': 10**np.linspace(-1,5,21),}
        f = GridSearchCV( SVC(kernel='rbf',gamma='auto'), params )
        f.fit(xtrain,ytrain)
        f.best_params_
        ```
        
        >  {’C’ : 12.589}
        
    

# [ 인공신경망 ANN ]

## 1. 퍼셉트론

- `퍼셉트론`
    - 인공신경망의 기본 단위로, 신경세포를 수학적으로 모델링한 것
    - 여러 입력을 받고, 이들을 합쳐서 특정 값을 넘으면 출력을 내보내는 간단한 구조
    - 여러 입력 값을 **선형적**으로 결합하여 가중치를 부여하고, **활성화 함수**를 통해 최종 출력을 결정

- `인공신경망`
    - 여러 퍼셉트론이 계층적으로 쌓여 구성
    - 입력 계층, 잠재 계층, 출력 계층으로 이루어짐
        
        ![2024-10-25_18-45-37.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/39bea4ed-196a-40ab-8521-fe761329167e/2024-10-25_18-45-37.jpg)
        

- `활성화 함수`
    - 활성화 함수에 따라 다양한 모델을 표현하는 것이 가능
        
        ![2024-10-25_18-46-28.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/6f5ef974-0f96-4c78-9e7c-59858e2a0c0e/2024-10-25_18-46-28.jpg)
        

- 파라미터 학습
    - 퍼셉트론은 일반적인 모델 → 모든 활성화 함수에 적용가능한 일반적인 방법이 필요
        
        → 손실함수 최소화
        
    - `손실 함수`
        - 해결하고자 하는 문제와 활성화 함수에 따라 손실 함수를 정의
        - 손실을 최소화하는 최적의 파라메터를 탐색
        - 경사하강법과 같은 일반적인 최적화 기법을 사용

## 2. 경사하강법

1. `경사하강법` (Gradient Descent)
    - 최적의 값을 찾기 위해 점진적으로 계산하는 방법 (밑으로 조금씩 내려가면서 목표 지점을 찾는 방식)
    - $min_{𝑤,𝑏}𝐿(𝑤, 𝑏)$ :
        1. 파라미터 초기화 
            - $𝑤^{(0)} = 𝑤_0, \quad 𝑏^{(0)} = 𝑏_0$
        2. 파라미터 업데이트
            - 설정해둔 학습률만큼, 미분된 각도로 이동
            - $w^{(k+1)} = w^{(k)} - \lambda \frac{\partial L}{\partial w} , \quad b^{(k+1)} = b^{(k)} - \lambda \frac{\partial L}{\partial b}$
            - $\lambda$ = 학습률
            - $\frac{\partial L}{\partial b}$ = 경사도
        3. 수렴할 때까지 반복
    - ex.
        
        ![2024-10-25_18-58-52.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/122d5cf8-56a3-4022-bbb6-6cd2f2889a62/2024-10-25_18-58-52.jpg)
        
        ![( 가로축 w, 세로축 b)](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/1af34372-487b-4938-9dd6-61e7ac0b874a/2024-10-25_18-57-48.jpg)
        
        ( 가로축 w, 세로축 b)
        
2. `확률적 경사하강법` (SGD, Stochastic Gradient Descent)
    - 일반적 경사하강법 : 경사도를 계산하기 위해 모든 데이터를 사용 → 계산량이 많음
    - SGD : `하나의 데이터` 이용 → 근사적으로 경사를 계산하는 방법
    - 장점
        - 손실을 바로 줄이지는 못하지만 더 빠르게 계산이 가능
        - 각 스텝은 부정확하지만 결국은 비슷한 곳으로 수렴
        
        ![2024-10-25_19-01-22.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/c381e6ee-df49-41c6-a772-b407ada4ae2d/2024-10-25_19-01-22.jpg)
        

1. `미니배치` (Mini-batch)
    - 배치(batch) : 훈련에 사용되는 한 셋의 데이터 (ex. 훈련집합)
    - 일반적 경사하강법 → 파라메터 업데이트 한 번에 전체 배치가 사용
    - 확률적 경사하강법 → 하나의 데이터로 업데이트
    - 미니배치 → 파라메터가 하나의 미니배치로 업데이트
        - 전체 훈련집합을 고정된 크기의 작은 집합 여러 개로 분할 ⇒ 미니 배치

- `학습률` (Learning rate)
    - 얼마나 빨리 파라메터를 업데이트 할지를 결정
    - 작은 $\lambda$ : 느리지만 안정적인 학습
    - 큰 $\lambda$ : 빠르지만 불안정한 학습 (수렴하지 않을 수도 있음)
        
        ![2024-10-25_19-16-28.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/77724385-a1a6-4f32-a979-a9919d099024/2024-10-25_19-16-28.jpg)
        

## 3. 소프트맥스 회귀

- 이진 분류 퍼셉트론
    - 로지스틱 회귀를 모델링 - 입력 X1과 X2를 통해 Y가 1 또는 0 값을 도출
    - `시그모이드 활성화 함수`를 사용해 Y가 1일 확률을 계산

- `One-hot encoding`
    - 기계학습에서 `범주형 변수를 표현`하는 기본적인 방법
    - 클래스별로 벡터의 특정 위치에 '1'로 표현
        
        ![2024-10-25_19-20-37.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/881f38ec-5f17-44e9-8204-f59f48ffc7cf/2024-10-25_19-20-37.jpg)
        

- `라벨링 문제`
    - One-hot encoding과 일반 이진 분류 퍼셉트론 분류를 사용하는 경우,
        - 𝑝1과 𝑝2를 비교하여 클래스를 결정
            
            ![2024-10-25_19-22-36.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/742a03fc-8a23-42a3-897a-648ca357cf42/2024-10-25_19-22-36.jpg)
            
            → 𝑝1 과 𝑝2 각각의 확률 계산 → `합이 1이 아님`
            
            ⇒ 손실 함수 적용 불가능
            

- `소프트맥스 회귀` (Softmax Regression)
    - 시그모이드 대신 소프트맥스 활성화 함수를 사용
    - Softmax activation :  $\sigma(z)_i = \frac{e^{z_i}}{\sum{k=1}^{K} e^{z_k}}$
    - 크로스 엔트로피 손실 :  $L = y_1 \log p_1 + y_2 \log p_2$
        
        ![2024-10-25_19-29-24.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/42962f69-b416-4ed3-8892-a59545587a39/2024-10-25_19-29-24.jpg)
        

- 소프트맥스는 멀티클래스 분류 문제에 쉽게 적용 가능
    
    ![2024-10-25_19-30-04.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/a10d6a4d-c7be-4762-b1f6-8556eff962f6/2024-10-25_19-30-04.jpg)
    

## 4. 다계층 퍼셉트론

- `다계층 퍼셉트론` (MLP)
    - 가장 기본적인 형태의 인공신경망
    - 모델 파라메터 : weights (경사하강법으로 훈련)
    - 튜닝 파라메터 : # of layer, # of nodes, activation, regularization
    - 학습 파라메터 : learning rate, batch size, and etc…

- 간단한 형태 (2단)
    - 히든 레이어 : 시그모이드 함수를 활성화 함수로 활용
    - 아웃풋 레이어 : 소트트맥스를 통해 이진 분류 수행
        
        ![2024-10-25_19-33-52.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/f6b90e01-e9e9-48a0-99f7-ce4c74926892/2024-10-25_19-33-52.jpg)
        

- 이진 분류
    
    ```python
    from sklearn.neural_network import MLPClassifier
    
    f = MLPClassifier(
            hidden_layer_sizes = (2,2),
            activation = 'logistic',
            solver = 'lbfgs', # for small data set, sgd/adam for large data set
            alpha = 0.001, # L2 regularization
            batch_size = 'auto',
            learning_rate = 'constant',
            learning_rate_init = 0.001,
            random_state = 0,
            max_iter = 10000)
    ```
    
    ```python
    f.fit(xtrain,ytrain)
    print( f.score(xtrain,ytrain), f.score(xtest,ytest) )
    ```
    
    >  0.509 0.497
    

- W
    
    ```python
    f.coefs_
    ```
    
    ![2024-10-25_22-13-55.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/b9275e66-aff7-4fbe-9479-9df949746418/2024-10-25_22-13-55.jpg)
    
    10x2 → 2x2 → 2x1
    

- +1
    
    ```python
    f.intercepts_
    ```
    
    ![2024-10-25_22-14-34.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/79f1690c-ca7a-4f1c-b615-80f40afd4fd4/2024-10-25_22-14-34.jpg)
