# [ 트리 모델 ]

## 0. 결정 나무 모델

- 의사 결정 나무(decision tree)
    - 전통적으로 실생활과 인공지능 양쪽에서 모두 흔히 사용되는 모델
        
        ![2024-10-25_23-50-25.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/00a84ceb-3c00-4bb6-bd64-9530a84ac398/2024-10-25_23-50-25.jpg)
        
    - 생성 방법
        1. 기존의 경험과 지식을 바탕 → 규칙기반의 AI
        2. 데이터 기반 → 결정나무모델
    
- `결정 나무 모델` (decision tree model)
    - 표본공간(Sample Space)을 분할하는 것과 동일
        
        ![2024-10-26_00-47-56.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/7ce628cb-4f6b-4455-bd96-d537b03aa0e0/2024-10-26_00-47-56.jpg)
        

- 각 집합의 `예측값` 할당?
    - 전체 에러 MSE를 최소화하는 예측값 = `각 영역 Y값의 평균`
        
        ![2024-10-26_00-49-24.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/c308749d-2fc4-4e34-885a-b34c10d006a9/2024-10-26_00-49-24.jpg)
        
    
- 분할 기준 설정? → 방법론 활용

## 1. 회귀 나무

- `회귀 나무` (Regression Tree)
    - Y가 `연속형 변수`일 때
    - **특징 공간**을 여러 개의 구간(노드)으로 나누고, 각 구간에서 **평균 값**을 예측하는 방식으로 작동
    
- `Recursive Binary Splitting` (RBS)을 이용하여 샘플공간을 분할
    1. 한 번에 `하나의 변수`를 이용하여 두 개의 공간으로 분할
        - 여러 변수 중 노드의 `평균 제곱 오차(MSE)를 최소화`하는 변수와 지점 선정
            
            ![2024-10-26_00-58-04.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/bed4fb5f-da67-40df-bb79-7c1bda137d1e/2024-10-26_00-58-04.jpg)
            
    2. 분할된 공간을 다시 같은 방식으로 분할
    3. 각 공간에 샘플이 하나가 남을 때까지 공간을 계속 분할할 수 있음
        - 나무의 크기 (종단 노드의 개수) 조절 가능

- 나무의 크기 (종단 노드의 개수)
    1. 최소 크기의 나무 ( |𝑇| = 1 ) : 모든 샘플을 하나의 값으로 추정 → Underfitting
    2. 최대 크기의 나무 ( |𝑇| = 𝑛 ) : n 개의 값으로 n 개의 샘플을 추정 → Overfitting
    
    ⇒ 나무의 크기 = 튜닝 파라미터로 조정 필요
    
- 최대 트리 모델
    
    ```python
    from sklearn.tree import DecisionTreeRegressor
    
    f = DecisionTreeRegressor(
        max_depth = None,       # 트리의 최대 깊이
        min_samples_split = 2,  # 분할할 최대 샘플의 수
        min_samples_leaf = 1,   # 하나의 노드가 갖는 최소 샘플 수
        max_leaf_nodes = None   # 최대 트리의 단말 노드 수
    )
    f.fit(xtrain,ytrain)
    print( f.get_depth(), f.get_n_leaves() )
    ```
    
    >  18 259
    

## 2. 분류 나무

- `분류 나무` (Classification Tree)
    - Y가 범주형 변수일 때
    - 각 공간의 주된 클래스를 이용하여 분류 및 확률 계산
        
        ![2024-10-26_01-00-46.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/e116bca2-9d4c-43dc-ba00-86d85f96c9a0/2024-10-26_01-00-46.jpg)
        

- 마찬가지로 RBS 사용 & `분할 지점 평가 방식의 차이` (MSE 대신)
    
    ![2024-10-26_01-01-35.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/9f05b55a-6a7d-4f56-9c92-99c7531e0158/2024-10-26_01-01-35.jpg)
    

## 3. CART

- `CART` (Classification and Regression Tree)
    - 회귀나무 + 분류나무

- 다른 모델과의 비교
    
    ![2024-10-26_01-03-23.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/bc4337ff-3957-467c-bd5d-e2a391eff467/2024-10-26_01-03-23.jpg)
    
    ⇒ 결정나무 모델의 최대 단점은 낮은 예측 성능
    

# [ 앙상블 모델 ]

- `앙상블 모델` (Ensemble Model)
    - 여러 모델을 합쳐서 하나의 모델을 구성
    - 다른 방식의 모델이거나 다른 데이터에서 훈련되었거나 혹은 둘 다
        
        ![2024-10-26_01-05-02.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/dd0b9948-ce83-470c-82d2-d5fc23111949/2024-10-26_01-05-02.jpg)
        

1. 전통적 앙상블 방법
    - 서로 다른 방식을 이용하는 몇 개의 모델을 합쳐서 구성하되 각 모델의 비중을 다르게 조절
    - ex. 스태킹
2. 현대적 앙상블 방법
    - 다른 데이터 셋에서 훈련된 동일한 많은 모델을 같은 비중으로 합쳐서 구성
    - ex. 배깅, 부스팅

## 1. 스태킹

- `스태킹` (Stacking)
    - 다수의 기본 모델의 출력을 입력으로 하는 새로운 메타모델을 구성
    - 각 모델들이 입력을 출력 y로 맞추기 위해서 학습
    - 기저 모델들의 `예측 결과를 모아서 **새로운 데이터셋**을 구성`하고, 이를 사용하여 메타 모델을 학습 후 수행
    - 모델을 여러번 중첩할 수 있으며, 레벨로 구분됨
    - ex. 레벨 1
        
        ![2024-10-26_01-21-56.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/78eacce9-0094-48b7-b046-19d285e963e8/2024-10-26_01-21-56.jpg)
        
- 스태킹 vs 다계층 퍼셉트론 (MLP)
    - 스태킹은 각 단계가 독립적으로 훈련되는데 비해서 MLP는 모든 계층이 한번에 훈련
    - MLP의 장점: 숨겨진 feature를 찾는데 유리
    - MLP의 단점: 훈련이 어려움, 항상 같은 단위모델이 사용됨

## 2. 배깅

- `배깅` (Bootstrap Aggregating)
    - 하나의 모델을 다수의 부트스트랩 집합에서 훈련하고 그 결과를 하나로 합쳐서 예측
    - `부트스트랩`: 원래의 훈련집합에서 `중복을 허용하는 재표집을 통해 구성`
    - 통합과정: 회귀문제는 `평균` & 분류는 투표
        
        ![2024-10-26_01-37-41.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/da2fb4d8-9945-40a7-89ad-df99f2e6418f/2024-10-26_01-37-41.jpg)
        
- 모델을 안정화시키는 강력한 기법으로, 의사나무와 같이 `불안정한 모델에 효과가 높음`
    
    ![2024-10-26_01-38-17.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/9650f630-65fc-44ed-9c2b-c5dfe154d46c/2024-10-26_01-38-17.jpg)
    
    → K-NN에서 낮은 K에서는 모델이 불안하기 때문에 효과가 높지만, 높은 K에서는 효과가 적음
    

- 배깅되는 모델의 수(B)에 따른 효과
    - 많은 수의 모델을 배깅하면 효과가 좋아지지만 계산량이 늘어남 & 적절한 수준 이후 `효과 증가 X`
    - 일반적으로 100~200개의 모델
        
        ![2024-10-26_01-42-02.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/65ab0b66-e371-427d-9cff-888784a2ec28/2024-10-26_01-42-02.jpg)
        

- `랜덤포레스트` (Random Forest)
    - 결정나무 모델은 안정적이지 않음 (어떤 데이터 셋으로 훈련하냐에 따라서 모델이 많이 달라짐)
        
        → 나무 모델 + 배깅 (배깅을 통해 결정나무 모델을 안정화)
        
        → 성능 증가, but 일부 주된 입력변수가 부트스트랩된 모델을 비슷하게 만드는 경향
        
    - 랜덤포레스트 모델
        - `결정나무모델 + 배깅 + Sub-spacing`
        - 공간을 분할할 때마다 임의로 선택된 일부 변수만을 이용하여 최적의 분할 방향과 지점을 선택
        - 일부 변수 : p개 중 $\sqrt{p}$ 개 (분류) & p/3개 (회귀)
            
            ![2024-10-26_02-05-43.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/5ad0100d-53ba-4597-aa97-a9696200fde4/2024-10-26_02-05-43.jpg)
            
        - 코드
            
            ```python
            from sklearn.ensemble import RandomForestRegressor
            
            f = RandomForestRegressor(
                n_estimators = 100,     # 앙상블할 트리의 수
                max_features = 0.33,    # 사용할 변수의 비율 (1이면 Sub-spacing 제거)
                max_depth = None,       # 트리의 최대 깊이
                min_samples_split = 2,  # 분할할 최대 샘플의 수
                min_samples_leaf = 1,   # 하나의 노드가 갖는 최소 샘플 수
                max_leaf_nodes = None,  # 최대 트리의 단말 노드 수
                random_state = 0        
            )
            f.fit(xtrain,ytrain)
            ```
            

## 3. 부스팅

- `부스팅` (Boosting)
    - 여러 개의 단순한 모델을 `점진적`으로 합쳐나가면서 복잡하게 만듦
    - 다수의 약한 모델을 합쳐서 하나의 강한 모델을 생성
        
        ![2024-10-26_02-08-48.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/e5001485-dbef-4d72-927b-ca2eb60c3ff2/2024-10-26_02-08-48.jpg)
        

1. `Adaptive Boosting` (AdaBoost)
    1. 모든 샘플을 동등하게 취급하여 하나의 간단한 모델을 학습
    2. 각 샘플의 에러를 측정
    3. 에러가 큰 샘플에 큰 가중치를 두어 또다른 간단한 모델을 학습
    4. 간단한 모델들을 합쳐서 하나의 복잡한 모델을 생성
    5. b~d를 반복하여 모델을 생성
        
        ![2024-10-26_02-11-11.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/481474b4-e61c-4222-98d5-e93736b87374/2024-10-26_02-11-11.jpg)
        

1. `Gradient Boosting Machine` (GBM)
    1. 주어진 데이터를 간단한 모델로 훈련
    2. 에러를 계산하고, 이 에러를 새로운 목표값으로 하여 다음 모델을 훈련
    3. 최종 에러가 충분히 작아질 때까지 a~b를 반복
        
        ![2024-10-26_02-12-39.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/18ed3688-4b66-410e-8b7a-4b32fb5e60ce/2024-10-26_02-12-39.jpg)
        
    - 코드
        
        ```python
        from sklearn.ensemble import GradientBoostingRegressor
        
        f = GradientBoostingRegressor(
            n_estimators = 100,     # 앙상블하는 트리의 수
            max_depth = 3,          # 트리의 최대 깊이
            min_samples_split = 2,  # 분할할 최대 샘플의 수
            min_samples_leaf = 1,   # 하나의 노드가 갖는 최소 샘플 수
            max_leaf_nodes = None,  # 최대 트리의 단말 노드 수
            random_state = 0        
        )
        f.fit(xtrain,ytrain)
        ```
        

- 부스팅되는 모델의 수(B)에 따른 효과
    - 너무 많은 모델을 사용하면 주어진 데이터를 과최적화하게 됨 - `U자형 커브`
