# [ 탐색적 데이터 분석 ]

- `탐색적 데이터 분석` (EDA, Exploratory Data Analysis)
    - 데이터를 이해하고 요약하는 과정
    - 데이터 분석의 첫 단계에서 데이터의 형태와 관계를 파악하는 데 도움
    - 주로 다른 가정 없이 진행
    - ex. 데이터 분포 확인, 시각화, 데이터 변형, 이상치 탐색, 결측치 처리, 통계 분석

## 1. 데이터 종류, 행렬, 형태

- 데이터의 종류
    1. `정형 데이터`
        - 일반적으로 표현되는 숫자, 범주 등의 데이터
        - 각 변수가 고유의 특징, 의미를 가지고 있음 → 데이터 행렬 형태로 표현
        - 통계, 기계학습, 딥러닝 등으로 분석 가능
        - ex. 키, 주소, 성별, ..
    2. `비정형 데이터`
        - 정형이 아닌 다른 모든 형태의 데이터
        - 각 변수의 의미를 찾기가 어려움 → 정형의 형태로 변환하여 사용 (정형화)
        - 분석이 어려웠지만 최근 딥러닝의 발전으로 분석이 가능해짐
        - ex. 이미지, 음성, 텍스트, ..

- 데이터 행렬
    - 행: 샘플, 표본, 개체 & 열: 변수, 피처, 항목
    - 변수의 수 (p) : 데이터 차원
    - 샘플의 수 (n) : 좁은 의미의 데이터의 크기 ↔ 넓은 의미 : n x p
        - 날씬한 행렬 (n>>p): 일반적이고 분석이 쉬움
        - 뚱뚱한 행렬 (n<p): 특수한 경우에 발생하고 분석이 어려움
            
            → 차원의 저주 (데이터의 차원이 높아질수록 분석이 매우 어려워짐)
            
        
        ![2024-10-22_16-28-12.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/b4de135b-0927-4841-a6b8-5c61c47521d0/2024-10-22_16-28-12.jpg)
        

- 데이터 형태
    1. `수치형 데이터` 
        1. `연속형` 데이터: 모든 실수값이 가능 (ex. 키, 몸무게)
        2. `이산형` 데이터: 특정 값(보통은 정수)만 가능 (ex. 메세지 수신 횟수)
            
            ⇒ 모두 연속형으로 분석
            
    2. `범주형 데이터`
        1. `명목형` 데이터: 순서가 없음 (ex. 성별, 지역, 오류코드)
        2. `순서형` 데이터: 순서가 있음 (ex. 상/중/하, 등급)
            
            ⇒ (일반) 모두 명목형으로 분석
            
            ⇒ (고급) 순서 고려하는 방법론 활용, 이산형 변환 등 변수에 대한 이해 활용
            
    - 구분
        - 숫자 형태인데 겹치는 값이 많을 때: 범주형 데이터에 대한 코드인지 확인 (ex. 1, 2만 반복되는 경우)

---

# [ 데이터 요약 ]

## 0. 데이터셋 준비

- 아이리스 데이터셋
    - 꽃잎의 길이와 너비를 바탕으로 아이리스(붓꽃)의 품종을 예측
    - 표본의 수: 150개 & 변수의 수: 5개
        - 수치형 : Sepal Length, Sepal Width, Petal Length,Petal Width
        - 범주형 : Species (3종류 : Setosa, Versicolor, Virginica)

1. 데이터셋 불러오기
    
    ```python
    import pandas as pd
    from sklearn.datasets import load_iris
    
    # 데이터셋 불러오기
    X, y = load_iris(return_X_y=True,as_frame=True)
    ```
    

1. 0, 1, 2로 되어있는 범주형 데이터 레이블 변환
    
    ```python
    id2label = ['setosa','versicolor','virginica']
    species = [id2label[i] for i in y]
    ```
    

1. 데이터 프레임 생성 및 변환
    
    ```python
    # 기존 데이터셋 복사
    data = X.copy()
    # 열 이름 변경
    data.columns = ['SepalLength','SepalWidth','PetalLength','PetalWidth']
    # Y에 해당하는 범주 데이터 열에 새로 추가
    data['Species'] = species
    ```
    
    ![2024-10-23_18-18-09.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/1057d39e-97bd-4721-bd15-816a634e49c5/2024-10-23_18-18-09.jpg)
    

## 1. 데이터 요약

- 대표값
    - 평균, 중앙값, 최빈값
        
        ![2024-10-22_17-13-30.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/9208ebcc-6643-42a0-bfaa-88d06197721e/2024-10-22_17-13-30.jpg)
        
        범주형 데이터 : 보통 최빈값 활용
        

- 분포
    - 분산 or 표준편차
        
        ![2024-10-22_17-11-13.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/241199ad-5715-4905-9500-81523eb4e7d6/2024-10-22_17-11-13.jpg)
        
    - `백분위수`
        - 25% Percentile (Q1): 아래에서부터 25%에 해당하는 값
        - 50% Percentile (Q2): 아래에서부터 50%에 해당하는 값
        - 75% Percentile (Q3): 아래에서부터 75%에 해당하는 값
        - 0%, 100% Percentile: 최소값, 최대값
    - 코드
        
        ```python
        data.describe()
        ```
        
        ![2024-10-23_18-27-56.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/9a0f5106-d51b-4bdc-aac1-ca6172a0cf12/2024-10-23_18-27-56.jpg)
        

- 수치형 데이터의 분포의 표현
    1. 도수분포표 : 데이터의 개수를 구간별로 나누어 표시한 표
    2. 히스토그램 : 연속형 변수의 도수분포표를 막대 그래프로 표현 : `data.hist()`
    3. 박스플롯 : 연속형 변수의 분포를 박스의 형태로 표현 : `data.boxplot()`
        
        ![2024-10-22_17-12-40.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/3f5620ab-6251-4acd-8010-d5ec0d2e8e5a/2024-10-22_17-12-40.jpg)
        

- 범주형 데이터의 분포의 표현
    1. 도수분포표 : 데이터의 개수를 구간별로 나누어 표시한 표
    2. 막대 그래프 : 범주형 변수의 도수분포표를 막대 그래프로 표현
    3. 원그래프 : 범주형 변수의 도수분포표를 원 그래프로 표현
    

## 2. 이변량 데이터 요약

- 이변량 데이터 요약
    - 두 데이터 사이의 관계에 대한 요약
    - 데이터 형태에 따른 요약 방법 (위: 시각화, 아래: 요약)
        
        ![2024-10-22_17-25-53.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/12850477-e14a-45d5-9b8d-5ab244119f69/2024-10-22_17-25-53.jpg)
        

### (1) 수치형 - 수치형

- 시각화
    1. 추세선 (trend line)
        - 두 변수 사이의 관계를 선의 형태로 표현
    2. `산점도` (scatter plot)
        - 두 변수의 값을 이차원 좌표 상에 점으로 표현
        - 두 수치형 변수 사이의 관계를 직관적으로 확인 가능
        
        ```python
        data.plot.scatter('SepalLength','SepalWidth')
        ```
        
        ![2024-10-23_19-09-44.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/4f93da5e-fd46-4d21-99f0-01e2937fb5e1/2024-10-23_19-09-44.jpg)
        

- 요약
    1. `공분산`
        - 두 데이터가 얼마나 같이 변하는지 나타내는 값
        - 단점 : 데이터의 스케일에 따라 값이 달라짐 (ex. 데이터 10배 → 공분산 10배)
            
            ![2024-10-22_17-27-58.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/a0ebfd29-c43e-4c06-9326-45ce040ed71d/2024-10-22_17-27-58.jpg)
            
    2. `상관계수`
        - 공분산의 분산으로 스케일한 값 (-1 ~ 1)
            
            ![2024-10-22_17-28-28.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/c56ad290-da27-4209-bcf9-ed7b7003dda0/2024-10-22_17-28-28.jpg)
            
        - 스케일에 영향 X
            
            ![2024-10-22_17-29-25.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/34cd2fad-4d59-4891-83a9-d84e8f5c05f3/2024-10-22_17-29-25.jpg)
            
        - 코드
            
            ```python
            data[['SepalLength','SepalWidth','PetalLength','PetalWidth']].corr()
            ```
            
            ![2024-10-23_19-09-14.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/d61c6297-1222-4dd6-abb6-714f72e1cdcb/2024-10-23_19-09-14.jpg)
            

### (2) 수치형 - 범주형

- 시각화 : `박스플롯`
    - 각 범주별로 수치형 데이터를 각각 시각화
    - 코드
        
        ```python
        data.boxplot('SepalLength',by='Species')
        ```
        
        ![2024-10-23_19-19-47.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/e00272d4-bc39-404c-9122-6e2335feb3cc/2024-10-23_19-19-47.jpg)
        

- 요약 : `SMD`
    - 범주형의 범주가 2개인 경우, 평균 차이 외에도 분산을 고려한 SMD (Cohen’s d)값 사용
        
        ![2024-10-22_17-32-53.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/36371739-b0eb-44be-ad36-8e4e27a76a1e/2024-10-22_17-32-53.jpg)
        
    - 코드
        
        ```python
        data.groupby('Species').mean()
        data.groupby('Species').var()
        ```
        
        ![2024-10-23_19-20-35.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/c65c82c0-a7c6-4222-93a2-88ada324db82/2024-10-23_19-20-35.jpg)
        
        ![2024-10-23_19-20-52.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/bfa6954e-3b24-484f-8c89-871145c27446/2024-10-23_19-20-52.jpg)
        
        ```python
        d = abs(5.006 - 5.936)
        s = ((50-1)*0.124249 + (50-1)*0.266433) / (50+50-2) )**(1/2)
        smd = d/s
        print(smd)
        ```
        
        >  2.10
        

### (3) 범주형 - 범주형

- 시각화 : `모자이크 플롯`
    
    ![2024-10-23_03-28-03.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/e6b65b06-eea4-49b9-867f-add13fedacb7/2024-10-23_03-28-03.jpg)
    
    → ABC의 비율이 각각 다름 → 관련성 존재
    
- 요약
    1. `크로스 테이블`
        - 코드
            
            ```python
            # 현재 데이터셋은 범주형이 하나여서 다른 데이터 설정
            pd.crosstab(data['sex'], data['smoker'])
            ```
            
            ![2024-10-23_19-26-17.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/5c7136e5-409d-43a5-9c30-9ba25c9b0e7f/2024-10-23_19-26-17.jpg)
            
    2. `승수비` (odd ratio)
        - “두 범주형 변수가 서로 관련이 있는가?“에 대한 수치적 값
        - 1에 가까울 수록 관련이 없고, 1에서 멀어질 수록 관련이 높음
            
            ![2024-10-23_03-31-19.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/0d2ee64c-3302-448d-83aa-b49b83a9aad6/2024-10-23_03-31-19.jpg)
            
            ![2024-10-23_03-30-54.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/5a457d12-f399-45c3-8e31-663c7ba93151/2024-10-23_03-30-54.jpg)
