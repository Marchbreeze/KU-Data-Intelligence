# [ 모델 선택 ]

## 1. 파라미터 종류

- 학습 모델의 파라미터
    1. `모델 파라메터`
        - 데이터로부터 `학습`되는 파라메터로 모델의 일부, 모델이 결정되면 자동으로 결정됨
    2. `하이퍼 파라메터` (튜닝 파라메터)
        - 데이터로부터 학습되지 않고 사용자가 `직접 지정`해주는 파라메터

## 2. 검증 집합의 분리

- `검증 집합`
    - 학습-평가 집합의 관계를 학습집합 내에서 재표집을 통해 시뮬레이션
        
        ![2024-10-25_04-57-04.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/5898c7b5-a631-442a-92b8-55fe65255362/2024-10-25_04-57-04.jpg)
        

- 모델의 복잡도(Complexity)에 따른 성능의 변화
    
    ![2024-10-25_04-57-36.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/130c94c5-353b-47e1-8b1d-80b7c2b549c5/2024-10-25_04-57-36.jpg)
    
    - 실제 모델 선택 과정에서 모델의 복잡성에 따라 훈련 집합과 평가 집합의 성능이 다르게 변할 수 있기 때문에 검증 집합 필요
    - 모델이 지나치게 복잡하면 훈련 데이터셋에 잘 맞더라도, 평가 데이터셋에서는 `오버피팅`이 발생하여 성능이 저하됨
        
        → 이를 피하기 위해 검증 집합을 통해 모델의 성능 패턴을 측정
        

- 검증집합은 실제 평가집합의 성능 패턴을 잘 보여줌
    - `검증집합을 통해 최적의 모델을 선택`하는 것이 가능
    - 하지만 검증 집합의 성능만으로 최종 성능을 파악하는 것은 위험하며, 검증 집합을 자주 보게 되면 이론적으로 모델이 검증 집합에 오버피팅될 가능성이 존재
        
        → 모델의 최종 성능은 반드시 `테스트 집합에서의 성능을 통해 측정`
        

## 3. 교차 검증

- **`교차검증`** (Cross-Validation)
    - 검증집합을 이용하는 방식과 유사하지만, `훈련집합과 검증집합이 서로 바뀔 수 있음`
        
        ![2024-10-25_05-20-37.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/c1b6e44a-ebf5-40c1-883a-c152b8583616/2024-10-25_05-20-37.jpg)
        

1. `K-폴드 교차검증` (K-Fold Cross-validation)
    - 전체 훈련집합을 K개의 그룹으로 나눈 후, 하나의 그룹을 검증집합 & 나머지를 훈련집합으로 검증 진행
        
        → 각각의 그룹에 대하여 K번 반복
        
    - K개의 검증집합의 성능을 평균하여 교차검증 성능을 측정
        
        ![2024-10-25_05-21-52.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/1b86595f-5b12-4554-b1f6-c103cf98de48/2024-10-25_05-21-52.jpg)
        

1. `LOOCV` ****(Leave One Out 교차 검증)
    - N개의 샘플에 대해 각 샘플을 테스트로 사용하여 N번 반복 평가하는 방식
    - 하나의 그룹이 하나의 샘플(데이터)만을 포함

⇒ 데이터의 수와 모델의 복잡도, 훈련 시간에 따라서 전략적으로 선택 (검증 집합, K-Fold, LOOCV)

## 4. K-NN 모델에서의 활용

- KNN (K-Nearest Neighbor) 모델
    - 예측하려는 지점 주변의 가장 가까운 K개의 샘플로부터 예측
    - K: 튜닝 파라미터
        
        ![2024-10-25_15-34-12.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/00b8b4b4-a3df-40c0-9e3c-5964c040dd05/2024-10-25_15-34-12.jpg)
        
    - 분류 :  K개의 가까운 샘플 중 가장 많은 클래스에 따라 예측 값을 결정
        - ex. K가 3일 때, 3개의 샘플 중 2개가 빨간색이면 해당 지점은 빨간색으로 분류
    - 회귀 : K개의 샘플의 Y값 평균으로 예측 값을 결정
    
- K-NN 회귀 모델
    
    ```python
    from sklearn.neighbors import KNeighborsRegressor
    
    f = KNeighborsRegressor(n_neighbors=5)
    f.fit(xtrain,ytrain)
    
    print('train accuracy: ', f.score(xtrain,ytrain))
    print('test accuracy: ', f.score(xtest,ytest))
    ```
    
    >   train accuracy:  0.59
    
    test accuracy:  0.36
    

- 교차검증 5번 진행 (K-fold)
    
    ```python
    from sklearn.model_selection import cross_val_score
    
    cv_score = cross_val_score(f,xtrain,ytrain,cv=5)
    print(cv_score)
    print(cv_score.mean())
    ```
    
    ![2024-10-25_15-54-15.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/45f3db3b-4102-4e8c-80be-f9c91265925e/2024-10-25_15-54-15.jpg)
    

- K 개수에 따른 성능($R^2$) 변화
    
    ```python
    for k in np.arange(1,100):
    	  f = KNeighborsRegressor(n_neighbors=k)
    	  f.fit(xtrain,ytrain)
    	  
    plt.plot(klist,r2_train,klist,r2_cv,klist,r2_test)
    plt.legend(['Train','CV','Test'])
    ```
    
    ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/d63271f2-40fa-4fc5-9936-b5b79ab4603f/image.png)
    
    - K가 커질수록 단순한 모델
        - K = 0 : 오버피팅 (훈련집합에 성능이 과도하게 맞춰짐)
        - K = 100 : 언더피팅
    - 교차검증의 성능이 가장 좋은 K= 16 의 모델을 선택

- 더 간단한 파라미터 튜닝 방법 (GridSearchCV)
    
    ```python
    from sklearn.model_selection import GridSearchCV
    
    params = {'n_neighbors': np.arange(1,100)}
    f = GridSearchCV( KNeighborsRegressor(), params, cv=5 )
    f.fit(xtrain,ytrain)
    
    f.best_params_
    f.best_score_
    ```
    
    >   {'n_neighbors': 16}
    
    0.45
    

---

# [ 모델 확장 ]

- 기본적인 모델들은 다양한 방식으로 확장이 가능
    - 우리가 사용할 수 있는 모델을 풍부하게 만듦
    - 일반적으로 `모델의 복잡성을 줄이는 방향으로 확장` → 오버피팅 가능성 낮춤

- 모델 확장 방법
    1. 입력 변수 선택 
        - 활용가능한 입력 변수 중 일부를 골라서 사용
        - 모든 모델에 적용 가능
    2. 차원 축소
        - 입력 변수의 차원을 더 낮은 차원으로 압축하여 사용
        - 연속형 입력 변수에 대하여 사용 가능
    3. 규제화
        - 모델 파라메터의 범위를 제한
        - 모델 파라메터가 존재하는 모수적 모델에 적용 가능
        

## 1. 입력 변수 선택 & Stepwise

- `입력 변수 선택` (Feature Selection)
    - 어떤 모델에 대하여 p개의 주어진 입력 변수 중 가장 좋은 k개의 변수를 선택

1. Best Selection
    - 모든 경우에 대하여 조사해보고 검증집합에서 가장 좋은 성능을 갖는 모델을 선택
    - p에 따라 경우가 기하급수적으로 증가하여 실질적으로 적용 불가능

1. `Stepwise Selection`
    - 한 번에 모든 조합을 찾는 것이 아니라, 그 전 단계에서 가장 좋은 조합으로부터 탐색을 시작
    1. Forward Stepwise Selection
        - 초기 모델에서 출발해 변수를 하나씩 추가하며 최적의 모델 선택
            
            ![2024-10-25_05-30-45.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/6f8a9a3f-da0a-4a5e-87dd-594f790772be/2024-10-25_05-30-45.jpg)
            
    2. Backward Stepwise Selection
        - 모든 변수를 포함한 채로 시작하여 성능이 떨어지는 변수를 제거
            
            ![2024-10-25_05-31-23.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/421d24ae-c97d-4431-b64b-881292101427/2024-10-25_05-31-23.jpg)
            
    - 활용
        - Forward와 Backward는 반드시 같은 결과를 내지 않음
        - Backward가 계산량이 많기 때문에 보통 Forward만 쓰거나 같이 씀

## 2. 차원 축소 & PCA

- `차원 축소` (Dimension Reduction)
    - 주어진 p차원의 입력 변수 공간 (X1, X2, …, Xp)를 더 낮은 k 차원의 공간(Z1, Z2, …, Zk)으로 변환
    - 특징
        - 차원의 저주를 피하기 위하여 (고차원 모델에서는 어떤 모델도 적합하지 않음)
        - 불필요한 정보를 제거 (예: X1과 X2 가 비슷한 경우)
        - 새로운 입력 변수(feature)를 추출
        - 시각화 (2, 3차원 공간만 시각화 가능)

- 방법
    1. `변수 선택` (Feature Selection): 주어진 변수 중 `일부를 선택`
    2. `변수 추출` (Feature Extraction): 원래 변수에서 `새로운 변수를 도출`

- `주성분 분석` (PCA, Principal Component Analysis)
    - 목적 : 적은 수의 변수로 많은 데이터 변화량을 설명
    - 데이터의 변화량을 가장 잘 설명하는 변수(주성분)를 차례로 추출
    - 좌표축 변환으로 생각할 수 있음
        
        ![2024-10-25_05-43-17.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/596f0fc5-c28d-44d0-8a41-c2a3f1089681/2024-10-25_05-43-17.jpg)
        
    - 차원 축소 시, 데이터 변화량을 적게 설명하는 변수들은 제거되어 효율적으로 차원이 축소
        - ex. 2차원 데이터를 1차원으로 축소 → 데이터의 약 50% 손실
    - z1 방향의 큰 변화량과 z2의 작은 변화량을 고려해 z2를 제거 → 차원 축소에도 대부분의 변화량을 유지
        
        → 주성분 분석을 통해 차원을 줄이면 데이터의 **정보 손실**을 최소화
        
    - 주성분 분석의 유용성은 이론적이며, 실제로는 검증을 통해 성능을 확인한 후 사용

## 3. 규제화 & L1, L2

- `규제화` (Regularization)
    - 모델 파라메터의 범위를 제한함으로써 모델을 복잡성을 줄이는 것이 가능
    - ex. 𝑌 ≈ 𝛽0 + 𝛽1𝑋     for −100 < 𝛽1 < 100
    
- 손실 함수에 규제화 적용 시
    - $L = \sum_{i=1}^{n} (y_i - \beta_0 - \beta_1 x_i)^2 + \lambda \beta_1^2$
        - $\lambda \beta_1^2$ : 벌점항 = $\lambda$Penalty($\theta$)
        
    - Loss (𝜃, 𝜆 | 𝑋, 𝑌) = Error (𝜃 | 𝑋, 𝑌) + 𝜆Penalty(𝜃)
        - $\lambda$ : 튜닝 파라미터 → 모델의 복잡도 조절 (작아지면 기존과 유사, 커지면  𝛽가 0에 가까워짐)
        - 𝜃 : 모델 파라미터
        - 에러 최소화와 함께 베타의 제곱에도 패널티를 부여하여, 베타를 제약함으로써 과적합을 방지

1. `L2 규제화`
    - $\text{Penalty}(\theta) = \theta_1^2 + \theta_2^2 + \dots + \theta_p^2$
    - 규제가 강해질 수록 각 파라미터가 0으로 접근 (하지만 0이 되지는 않음)
    - `Ridge Regression`
        - $L = \sum_{i=1}^{n} \left( y_i - \beta_0 - \sum_{i=1}^{p} \beta_i x_i \right)^2 + \lambda \sum_{i=1}^{p} \beta_i^2$
            
            ![2024-10-25_05-54-19.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/4702a5d6-ce74-4b43-ad87-72e09016c5a1/2024-10-25_05-54-19.jpg)
            
            - **λ** 값이 커질수록 모델은 간단해지며, 훈련 데이터셋에서는 성능이 저하될 수 있음
            - 검증집합에서는 λ 값에 따라 U자형 커브가 나타나며, 이를 통해 적절한 모델을 선택할 수 있음
        - 코드
            
            ```python
            from sklearn.linear_model import Ridge
            
            f = Ridge(alpha=1)
            f.fit(xtrain,ytrain)
            print( f.coef_ )
            print( f.score(xtrain,ytrain), f.score(xtest,ytest) )
            ```
            
            ![2024-10-25_16-17-06.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/2c54cf41-649d-4356-8b3e-61183f69f34a/2024-10-25_16-17-06.jpg)
            
        
2. `L1 규제화`
    - $\text{Penalty}(\theta) = |\theta_1| + |\theta_2| + \dots + |\theta_p|$
    - 규제가 강해질 수록 각 파라메터가 0으로 접근하다가 0이 됨 → `변수 선택 가능`
    - `Lasso Regression`
        - $L = \sum_{i=1}^{n} \left( y_i - \beta_0 - \sum_{i=1}^{p} \beta_i x_i \right)^2 + \lambda \sum_{i=1}^{p} |\beta_i|$
            
            ![2024-10-25_05-56-50.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/a6507558-d0d3-4415-9a63-5cab2c85cbc1/2024-10-25_05-56-50.jpg)
            
    - 코드
        
        ```python
        from sklearn.linear_model import Lasso
        
        f = Lasso(alpha=1)
        f.fit(xtrain,ytrain)
        print( f.coef_ )
        print( f.score(xtrain,ytrain), f.score(xtest,ytest) )
        ```
        
        ![2024-10-25_16-25-13.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/2ca3204c-7647-4bf5-a0bb-d81688a368be/2024-10-25_16-25-13.jpg)
        

1. Elastic Net : L1과 L2 규제를 동시에 사용하는 방법
2. L0 규제화 :  0이 아닌 파라미터의 수를 찾는 것으로, 변수 선택과 동일
