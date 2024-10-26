- 선형 모델 : 모집단에서의 실제 𝑓()가 선형성을 갖고 있다고 가정
    - 선형 회귀 (linear regression) 모델: 회귀 문제에 적용 - 수치형 데이터
    - 로지스틱 회귀 (logistic regression) 모델: 분류 문제에 적용 - 범주형 데이터

# [ 선형 회귀 모델 - 수치형 ]

## 1. 단순 선형 회귀

- `단순 선형 회귀` (Simple Linear Regression)
    - 하나의 독립변수(X)와 하나의 출력변수(Y)에 대한 모델
        
        ![2024-10-25_01-02-24.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/ae3efe44-4409-436a-a640-029eef306344/2024-10-25_01-02-24.jpg)
        
    - 각 데이터의 모델링
    - 실제 데이터의 오류나 누락된 정보를 반영하기 위해 에러 항을 추가
        
        ![2024-10-25_01-02-34.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/59820936-abf8-48fe-8a3b-ed614ddff6c5/2024-10-25_01-02-34.jpg)
        
        ![2024-10-25_01-02-10.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/71547670-059a-4653-bdee-2558aa90d2fe/2024-10-25_01-02-10.jpg)
        

- 파라미터 추정
    - 전체 모집단을 직접 관찰할 수 없으므로, 랜덤하게 수집한 데이터를 이용해 훈련 데이터셋을 만들어야 함
        - 훈련 데이터에는 n개의 x와 n개의 y가 존재
        - 추정된 값인 $\hat{β_0}$과 $\hat{β_1}$은 훈련 데이터셋에 따라 달라지며, 진짜 파라미터와 완전히 일치하지 않음

- `손실 함수` (Loss Function)
    
    ![2024-10-25_01-20-11.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/0b374ba4-9a4a-4d1c-8246-634ca904e109/2024-10-25_01-20-11.jpg)
    
    을 미분하면
    
    ![2024-10-25_01-20-29.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/2a1b40f2-f014-46a7-b4ec-85ba6031c446/2024-10-25_01-20-29.jpg)
    

- 추정값은 훈련 데이터의 에러를 최소화하는 것이지, 반드시 평가 데이터의 에러를 최소화하지는 않음
    
    → 평가 데이터의 에러를 최소화하기 위해 모델 선택 과정이 필요
    

## 2. 일반 선형 회귀

- `일반 선형 회귀` (Multiple Linear Regression)
    - 하나의 출력 변수와 `여러 개의 입력 변수`
        
        ![2024-10-25_01-35-23.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/fbf22a9c-ec8f-4784-bc39-8587172c252d/2024-10-25_01-35-23.jpg)
        
    - 제곱 오차 손실 함수 최소화하여 파라미터 추정 진행
        
        ![2024-10-25_01-35-51.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/8d0b8093-ae1d-4096-8b97-405f24421f09/2024-10-25_01-35-51.jpg)
        
    - 특정 변수 (x1)이 1만큼 변할 경우, y는 β만큼만 변화
        
        → `각 X는 독립적으로 Y에 영향`, 귀 모델은 변수의 수에 관계없이 적용할 수 있음
        

## 3. sklearn 활용

- 훈련 데이터와 평가 데이터를 임의로 분할
    
    ```python
    from sklearn.model_selection import train_test_split
    
    xtrain, xtest, ytrain, ytest = train_test_split(X,y,test_size=0.4,random_state=42)
    ```
    

- 회귀 진행
    
    ```python
    from sklearn.linear_model import LinearRegression
    
    f = LinearRegression()
    f.fit(xtrain,ytrain)
    print( f.intercept_, f.coef_ )
    ```
    
    ![2024-10-25_03-36-26.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/41a85efe-12e5-4d31-8cca-ab23fde1f566/2024-10-25_03-36-26.jpg)
    

## 4. 회귀 모델 평가

- 추정한 모델이 얼마나 좋은지 평가하는 척도

1. `평균제곱에러` (MSE: Mean Square Error)
    
    ![2024-10-25_01-38-26.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/e563c1e3-a3d4-4654-a183-b48eaf2c6750/2024-10-25_01-38-26.jpg)
    
    - MSE는 스케일의 문제로 인해 해석이 어려울 수 있음 → `루트 평균 제곱 오차(RMSE)`로 변환하여 사용

1. `결정계수` $R^2$ (coefficient of determination)
    
    ![2024-10-25_01-39-14.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/d55dc5f2-ab34-4ee1-82a6-3204115336f0/2024-10-25_01-39-14.jpg)
    
    - Y가 얼마나 X에 의해 설명되는지에 대한 비율
    - 1 - (설명되지 않는 변동성 / 전체 변동성)
    - 0에서 1 사이의 값을 가지며, 값이 클수록 모델이 더 잘 설명하고 있다는 의미 (아주 나쁘면 음수도 가능)

- 회귀 분석 평가
    
    ```python
    f.score(xtrain, ytrain)   # 훈련 데이터에 대한 R2
    f.score(xtest, ytest)     # 평가 데이터에 대한 R2
    ```
    
    >   0.36
    
    0.29
    

## 5. 선형 회귀 모델의 확장

- 범주형 변수
    - 입력 변수가 범주형 변수인 경우 → `가변수`(dummy variable)로 변환하여 모델링
        - ex. 성별: X1 = 1 (남성) or 0 (여성)
    - 일반적으로 K개의 범주를 갖는 변수에 대해서 K-1 개의 가변수가 필요
        - ex. 지역: $X_{3S}$ = 1 (서울) or 0 (서울 외), $X_{3K}$ = 1 (경기) or 0 (경기 외)

- 변수 간 상호작용
    - 상호작용을 고려한 모델: 두 변수의 영향력이 비독립적
        
        ![2024-10-25_01-46-21.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/16980d31-ed45-481a-9330-732dcb03d364/2024-10-25_01-46-21.jpg)
        
        → `상호작용 항 𝑋1𝑋2` 를 새로운 변수 𝑋3 처럼 취급
        

- 다항 회귀
    - 다항 회귀 (Polynomial Regression) - 곡선형
        
        ![2024-10-25_01-47-18.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/138ef5d1-00a8-41fd-8ca1-2376f734c654/2024-10-25_01-47-18.jpg)
        
    - 차항을 추가하여 비선형성을 모델링 → 모델의 복잡도를 높이는 동시에 오버피팅의 위험성 증가

---

# [ 로지스틱 회귀 모델 - 범주형 ]

## 1. 이진 분류

- `이진 분류` (Binary Classification)
    - 주어진 입력 데이터를 기반으로 해당 데이터가 두 개의 카테고리 중 하나에 속하는지 예측

- `시그모이드 함수`
    - 주어진 실수 값을 `0과 1 사이의 확률 값으로 변환`해 주는 함수
    - 입력 값이 작을수록 결과는 0에 가까워지고, 입력 값이 클수록 결과는 1에 가까워짐
    - ex. 시그모이드 함수의 결과 0.7 → 해당 데이터가 클래스 1에 속할 확률 70%
    - $\hat{y} = \sigma(z) = \frac{1}{1 + e^{-z}}$

- `로지스틱 회귀` (Logistic Regression)
    - 클래스에 대한 확률을 시그모이드 함수를 이용하여 모델링
        
        ![2024-10-25_02-49-54.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/f9fdf951-e48b-4311-bc30-273c89e543ad/2024-10-25_02-49-54.jpg)
        
    - 로짓을 회귀한 회귀 모델
        - odds와 logit을 계산하면 이를 통해 선형 회귀와 유사한 형태의 회귀 모델이 도출
        
        ![2024-10-25_02-52-04.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/30ab1ce8-693b-46ab-9ff5-cb3ce15b8d09/2024-10-25_02-52-04.jpg)
        
    - 코드
        
        ```python
        from sklearn.linear_model import LogisticRegression
        
        f = LogisticRegression()
        f.fit(xtrain_simple,ytrain)
        print( f.intercept_, f.coef_ )
        ```
        
        >  [-8.20] [[1.34]]
        

## 2. 최대우도법

- `우도` (Likelihood)
    - 어떤 모델을 가정했을 때 현재 데이터를 관측할 확률
        
        ![2024-10-25_02-59-46.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/ec3f83f1-7e1a-462b-9fbc-27be09749949/2024-10-25_02-59-46.jpg)
        
        - **모든 데이터의 확률의 곱**
    - ex.
        
        ![2024-10-25_03-03-05.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/0a4ee23d-b42d-42d7-92e8-30184eda9a65/2024-10-25_03-03-05.jpg)
        
        $l(\beta_0, \beta_1) = \frac{1}{1 + e^{\beta_0 + \beta_1 \times 0.50}} \times \frac{e^{\beta_0 + \beta_1 \times 3.30}}{1 + e^{\beta_0 + \beta_1 \times 3.30}} \times \frac{e^{\beta_0 + \beta_1 \times 1.75}}{1 + e^{\beta_0 + \beta_1 \times 1.75}} \times \frac{1}{1 + e^{\beta_0 + \beta_1 \times 3.00}}$
        

- `최대우도법` (Maximum Likelihood)
    - 파라미터의 추정 방법
    - 우도의 미분을 통해 파라메터를 찾는 것은 계산이 어려움
        
        → 우도의 로그값(Log-likelihood)을 최대화하는 파라메터를 찾음
        
        → 풀 수 없어 수치적으로 풀어야 함 (뉴턴 랩슨 법이나 경사하강법과 같은 알고리즘 이용)
        

- 일반 로지스틱 회귀
    - 다수의 변수가 있을 때도 로지스틱 회귀 모델을 사용할 수 있으며, 기존의 선형 회귀 모델처럼 변수들을 추가하여 문제를 해결할 수 있음

## 3. 분류 모델 평가

- `혼동행렬`
    
    ![2024-10-25_03-10-56.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/0f92cb74-8536-4d5e-bb4f-706f8ff0b970/2024-10-25_03-10-56.jpg)
    
    - **정확도(Accuracy)** : $\frac{TP + TN}{TP + FP + FN + TN}$ → `score`
    - **재현율(Recall)** : $\frac{TP}{TP + FN}$ = 실제 positive 데이터 중, 맞게 예측한 비율
    - **정밀도(Precision)** : $\frac{TP}{TP + FP}$ = 예측 positive 데이터 중, 맞게 예측한 비율
    - **위양성률(False Positive Rate)** : $\frac{FP}{FP + TN}$ = 실제 negative 데이터 중, 잘못 예측한 비율
    - **F1 Score** :  $2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$
    
- Precision-Recall curve
    - recall과 precision은 반비례 (보통의 경우)
        - `mAP` (mean Average Precision): 단조감소화한 PR 커브의 면적
    - ROC curve: FP의 비율 증가함에 따라 TP의 비율은 항상 같거나 증가
        - `AUC` (area under curve): ROC 커브의 면적
        
        ![2024-10-25_03-16-35.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/e5810d9d-0eb9-4ef5-b99e-bafb7207e1c0/2024-10-25_03-16-35.jpg)
        
        mAP & AUC
        

## 4. 멀티 클래스 분류

- 3개의 클래스의 경우,
    
    𝑌 = 𝐴, 𝐵, or 𝐶에 대하여, A vs C와 B vs. C로 나누고 확률을 계산
    
    Pr 𝑌 = 𝐴 𝑋 + Pr 𝑌 = 𝐵 𝑋 + Pr 𝑌 = 𝐶 𝑋 = 1
