## 0. 군집화

- `군집화` (Clustering)
    - 데이터 집합에서 하위 그룹(subgroups) 또는 군집(cluster)을 찾는 매우 광범위한 방법론
    - 샘플 간의 유사성을 측정하기 위해 거리나 **유사도** 개념을 사용
    - n개의 샘플을 적당히 소수의 그룹으로 묶어 분석의 복잡성을 줄임
    - `비지도 학습`

## 1. K-means 군집화

- 중심점 (Centroid)
    - 군집의 중심점
- `군집 내 분산 (WCV)`
    - 동일한 군집 내 표본의 분산 - 평균으로부터 중심점까지 거리 활용

- `K평균 군집화`
    - 군집 내 분산(WCV)을 최소화하는 K개의 군집으로 분류
    - $𝐾^𝑛$ 가지 방식에 대한 조사가 필요 → 어려움
    - 차선해 알고리즘
        1. 샘플을 무작위로 K개의 군집에 할당
        2. 중심점 계산
        3. 샘플을 각각 가장 가까운 중심점의 군집에 할당
        4. 2~3을 수렴할 때까지 반복
    - 무작위 초기화로 인해 서로 다른 군집화 결과가 나타날 수 있음
        
        → 보통 여러 번 반복 진행 & 더 작은 WCV를 갖는 군집화를 선택
        

- `군집의 개수` = 튜닝 파라미터
    - 많은 군집 개수: 낮은 WCV, 낮은 재현성, 설명이 어려움
    - 적은 군집 개수: 높은 WCV, 높은 재현성, 설명이 쉬움
    - `Elbow method` (WCS 를 많이 감소시키지 않는 K 를 선택)
        
        ![2024-10-26_03-34-28.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/3ad2a637-e783-4819-aa4c-c08ab9cb5b3f/2024-10-26_03-34-28.jpg)
        

- 중요 이슈
    1. `스케일링` : 군집화는 주로 분산이 큰 변수에 의해 결정 → `정규화`가 필요한 경우도 존재
    2. `고차원` : 고차원 공간에서 한 점은 모든 다른 점으로부터 멀리 떨어져 있음 → 차원 축소 고려
    3. `거리 측정` : K평균 군집화는 일반적으로 유클리드 거리만 사용함 → 맨해튼 거리 사용시 K-medians
    4. `범주형 변수` : 거리측정이 불가 → 가변수화 & 정규화

- 코드
    - 정규화 전
        
        ```python
        from sklearn.cluster import KMeans
        
        f = KMeans(n_clusters=3,random_state=0)
        f.fit(X)
        pd.crosstab(f.labels_,y)
        ```
        
        ![2024-10-26_04-03-48.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/a7003654-83e8-4066-8313-0cd3d452729c/2024-10-26_04-03-48.jpg)
        
    - 정규화 후
        
        ```python
        from sklearn.preprocessing import StandardScaler
        
        scaler = StandardScaler()
        scaler.fit(X)
        Xscaled = scaler.transform(X)
        
        f2 = KMeans(n_clusters=3,random_state=0)
        f2.fit(Xscaled)
        pd.crosstab(f2.labels_,y)
        ```
        
        ![2024-10-26_04-07-25.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/94cd7023-ef40-4aaf-8fab-4a7aa1d33b4c/2024-10-26_04-07-25.jpg)
        
    

## 2. 가우시안 혼합 모델

- `가우시안 혼합 모델` (GM, Gaussian Mixture)
    - K 평균 군집화와 유사하지만, 데이터가 여러 클러스터에 속할 `확률`에 기반하여 보다 유연하게 분석
    - 소프트 군집화 (각 군집에 대한 확률 할당)
    - 특정 지점이 각 클러스터에서 나올 확률을 비교하여 데이터가 어느 클러스터에 속하는지 결정
    - 각 군집의 속할 확률과 그 군집에서 나올 확률을 모두 곱하여 실제 데이터를 관측할 확률을 산출
        
        ![2024-10-26_03-51-36.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/dfc13ba2-9f18-41af-ad4e-6b66bfcf948e/2024-10-26_03-51-36.jpg)
        
    - 표본이 할당된 군집의 `정규 분포`에서 생성되었다고 가정
        
        ![2024-10-26_03-54-53.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/6d481013-b780-4d43-a386-d32ff191a86f/2024-10-26_03-54-53.jpg)
        

- 수학적 모델링 (K=2인 경우)
    - 표본 i를 관측할 확률
        
        ![2024-10-26_03-52-43.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/7a69f1a2-e63c-4de2-a385-ac36bb71551a/2024-10-26_03-52-43.jpg)
        
        각 군집의 속할 확률과 그 군집에서 나올 확률을 모두 곱하여 실제 데이터를 관측할 확률을 산출
        
    - 최대 우도(maximum likelihood)를 사용 → 데이터 관측 확률을 최대화

- `Expectation-Maximization` (EM) 알고리즘
    1. 파라미터 초기화
    2. E-step: 주어진 파라미터로 할당 확률을 계산하여 군집화
    3. M-step: 현재의 군집으로 우도를 최대화하는 파라미터 계산
    4. 반복
        
        ![2024-10-26_03-56-02.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/2a3ee256-0337-473e-8316-1f23a24e3d8c/2024-10-26_03-56-02.jpg)
        

- GM의 장점
    - 스케일링에 영향을 받지 않음
    - 군집별로 확률을 계산

## 3. 계층적 군집화

- `계층적 군집화` (Hierarchical Clustering)
    - 군집들 사이에 계층 구조가 형성 → 나무 구조의 `덴드로그램`(dendrogram)으로 표현

- 덴드로그램의 해석
    - 말단(leaf)은 하나의 표본에 해당
    - 가장 유사한 두 샘플이 함께 묶여 하나의 군집을 형성
    - 노드에서 연결된 링크의 높이는 유사도를 표현
    - 덴드로그램을 적절한 높이에서 잘라 원하는 K개의 군집을 형성
        
        ![2024-10-26_03-57-44.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/815ede8d-706b-4c3f-87f4-ae5fb9afe9c3/2024-10-26_03-57-44.jpg)
        
    - ex.
        
        ![2024-10-26_03-58-22.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/282f2f4c-5538-4dba-88fb-d718ca3a5dd2/2024-10-26_03-58-22.jpg)
        
        - 5 와 7 은 가장 유사도가 높음
        - 9 와 2 는 유사도가 낮음 & 9 는 8, 5, 7과 유사한 것 만큼 2에 유사함
        - 수평적으로 가까움은 반드시 높은 유사성을 의미하지는 않음

- 알고리즘
    - 모든 군집 쌍에 대하여 거리를 측정 → 가장 거리가 가까운 두 군집을 식별하여 하나의 군집으로 병합

- Linkage : 군집-군집 사이의 거리 측정법
    1. Complete: 표본들 사이의 최대 거리
    2. Single: 표본들 사이의 최소 거리
    3. Average: 표본들 사이의 평균 거리
    4. Centroid: 중심점 사이의 거리

- K평균 군집화 대비 장점
    - 다른 수의 군집을 찾기 위해 다시 계산할 필요 없음 (K개의 군집 : 적절한 높이에서 트리를 자름)
    - 덴드로그램을 통해 적절한 군집의 수를 예상하는 것이 가능

- 코드
    
    ```python
    from sklearn.cluster import AgglomerativeClustering
    
    f = AgglomerativeClustering(
        n_clusters=3,
        metric='euclidean',
        linkage='complete'
      )
    f.fit(X)
    
    pd.crosstab(f.labels_,y)
    ```
    
    ![2024-10-26_04-14-38.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/35318f60-07d5-4982-93a6-930f1204ad31/2024-10-26_04-14-38.jpg)
