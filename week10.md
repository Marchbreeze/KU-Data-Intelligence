## 1. 언어인지 개요

- 언어 인지 : 사람이 사용하는 언어를 이해하고 해석하는데 중점
- 자연어처리 (NLP)를 활용 - 언어에 관련된 다양한 업무를 인공지능을 통해 해결 (음성 인지는 독립적)
    1. 감성분석 : 글의 분위기 파악 (문장 분류)
    2. 질의응답 : 주어진 정보를 바탕으로 질문에 대답
    3. 작문 : 주어진 키워드, 사실 관계를 바탕으로 새로운 글을 작성
    4. 번역, 요약, 채팅, 프로그래밍

### (1) TF-IDF

- 딥러닝 이전의 NLP
- TF-IDF (Term-Frequency and Inverse Document Frequency)
    - 문서 특성을 추출하여 문장을 표현하는 데 사용된 방식
    - 단어 빈도 활용, 순서와 관계 무시
    - 비정형 텍스트를 정형 데이터로 변환하여 다양한 분류 및 기계학습 문제를 해결

- 문서 내 단어의 가중치를 설정하여 특성을 표시 - 특정 단어의 반복 또는 독특함에 따라 가중치가 달라짐
    
    ![2024-12-19_01-37-21.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/29773ee7-ed65-4597-8e26-7bca6191d9a9/2024-12-19_01-37-21.jpg)
    
    - ex. apple : 문장 0에만 존재 → 문장 0을 특정짓는 단어 (가중치 높음)

### (2) One-hot Encoding

- One-hot encoding
    - 딥러닝 도입 이후, 각 단어를 unit vector로 변환하여 컴퓨터가 이해할 수 있는 형식으로 문장을 표현
    - 하나의 문서는 1과 0으로 이루어진 희소 행렬로 표현 (전체 행렬에서 0이 대부분을 차지)

- ex. ‘I love you, you love me’ : 4 (# of words) x 6 (text length) 행렬로 표현
    
    ![2024-12-19_01-41-00.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/3f0fd135-c10c-49dd-89a4-3787e1d3bd1e/2024-12-19_01-41-00.jpg)
    
- 문제점
    1. 너무 희소(sparse)하여 데이터 공간이 낭비됨
    2. 단어 사이의 관계를 내포하지 못함
        - “I"와 "me"는 의미적으로 유사함에도 불구하고, 벡터 상의 거리 차이는 "I"와 “love”의 거리와 동일

### (3) Embedding

- 임베딩 (Embedding)
    - 고차원에서 One-hot encode된 토큰의 벡터 → 저차원의 잠재공간(latent space)으로 매핑
    - N (# of tokens) → p (# of hidden features)
    - N x p의 임베딩 행렬이 필요 (보통 데이터로부터 학습됨)

- 차원 축소 - 단어 간의 **유사** 관계를 최대한 유지
    1. 4차원의 One-hot encode
        
        ![2024-12-19_01-45-10.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/1610a71c-5159-4e0d-9f9d-f578f91bb2c4/2024-12-19_01-45-10.jpg)
        
    2. 2차원의 잠재공간으로 축소 (유사 관계 유지하며)
        
        ![2024-12-19_01-45-51.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/b5db6a86-c871-4211-8d39-94f5dd27b28a/2024-12-19_01-45-51.jpg)
        

- 잠재공간(Latent Space)
    - 각 단어의 의미를 내포하는 잠재 변수(latent vector)로 이루어진 저차원의 공간
    - 비슷한 단어들이 가까이 위치하는 것으로 나타내며, 이는 학습을 통해서 변환 행렬을 찾아내어 구현
        
        ![2024-12-19_01-47-08.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/76f86b18-a8e3-4c8c-8259-b7765c512a6f/2024-12-19_01-47-08.jpg)
        

- 텍스트 데이터는 잠재 공간에 맵핑된 토큰의 시퀀스로 표현
    - 문장이 길어질수록 벡터의 시퀀스 또한 길어짐
    - 문장의 길이가 다름 → **순차적인 입력**을 받을 수 있는 모델이 필요
    - 문장은 일정한 너비를 갖지 않고, 순서를 가지고 있어 이미지처럼 고정된 길이로 표현할 수 없음
        
        ![2024-12-19_01-48-08.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/286f97f8-c2ff-47fe-9fdf-cc1f6eda262c/2024-12-19_01-48-08.jpg)
        
        - y 값은 과연 이러한 문장이 긍정적인 문장이냐, 부정적인 문장이냐를 판단하는, 문장 분류 진행

## 2. 순환신경망 (RNN)

- 순환신경망 (RNN, Recurrent Neural Network)
    - 순차 데이터를 처리하는 신경망 모델, 시계열 데이터 분석에도 사용 가능
    - 입력을 반복적으로 처리하며, 입력의 **히든 정보**가 다음 입력에 연결되어 최종 출력으로 이어짐
    - RNN cell이 데이터 길이만큼 반복되어 사용되는 형태
        
        ![2024-12-19_02-21-00.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/00da7b88-4d25-41d9-9460-92ffa9fd2feb/2024-12-19_02-21-00.jpg)
        

- 은닉정보(ℎ, hidden output)
    - 각 순차별 셀의 출력으로 최종 출력(𝑦)과는 구별
    - 셀 모델 자체는 모든 시점에 동일하지만 입력(현재 입력 + 과거 셀의 은닉정보)에 따라 현재 셀의 은닉정보가 결정됨
        
        ![2024-12-19_02-26-52.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/151b82e9-222f-424b-8311-6a0c96d13444/2024-12-19_02-26-52.jpg)
        

- RNN을 사용한 NLP모델
    - 문장 시작과 끝에 특정 토큰을 사용하여 각 단어를 차례로 처리하며 긍정, 부정 결과를 도출하는 구조
        
        ![2024-12-19_02-20-38.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/bf205bad-2162-472a-81dc-1c37d3650ca6/2024-12-19_02-20-38.jpg)
        
        - ex. 작 토큰을 -1, 끝 토큰을 1로 설정한 후 결과값 분류
        
- 문제에 따라 다양한 형태가 존재
    - 입력은 순차적이지만 최종 출력은 단일일 수도 있음
    - 경우에 따라 문장 형태의 여러 출력이 시퀀스 형태로 만들어질 수도 있음
        
        ![2024-12-19_02-24-23.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/1fa98f62-7c8d-4776-82bb-844021e79364/2024-12-19_02-24-23.jpg)
        

### (1) Simple RNN

- Simple RNN (Naïve RNN)
    - 가장 기본적인 RNN 구조
    - 일반적으로 하나의 단일 계층 신경망을 사용
    - 활성화 함수 : 하이퍼 탄젠트 → 경사도 소멸을 완화
        
        ![2024-12-19_02-29-06.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/e43b921a-0c37-4e62-989e-222b3df28769/2024-12-19_02-29-06.jpg)
        

- 히든 정보 계산 방법
    
    ![2024-12-19_02-29-47.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/452e5b8d-941d-48e6-b620-30cbb2d5f00e/2024-12-19_02-29-47.jpg)
    
    - 은닉벡터 크기 2, 입력벡터 크기 2인 경우
    - 파라미터의 수 : $2*(2+2+1) = 10$
    - $ℎ_{𝑡,1}= tanh(𝑤_{ℎ,11}ℎ_{𝑡−1,1} + 𝑤_{ℎ,12}ℎ_{𝑡−1,2} + 𝑤_{𝑥,11}𝑥_{𝑡,1} + 𝑤_{𝑥,12}𝑥_{𝑡,2} + 𝑏1)$
    - $ℎ_{𝑡,2}= tanh(𝑤_{ℎ,21}ℎ_{𝑡−1,1} + 𝑤_{ℎ,22}ℎ_{𝑡−1,2} + 𝑤_{𝑥,21}𝑥_{𝑡,1} + 𝑤_{𝑥,22}𝑥_{𝑡,2} + 𝑏1)$
        
        ![2024-12-19_02-32-40.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/0ea02eee-3447-494f-97b7-e180871334d7/2024-12-19_02-32-40.jpg)
        

- 최종 출력(y) : 은닉정보(h)를 활용해 다시 계산
    
    ![2024-12-19_02-33-17.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/80cbed62-6194-43dc-a2ae-e0a01dbcf357/2024-12-19_02-33-17.jpg)
    
    - 문장 분류의 경우 - 마지막 셀에서 출력 계산
    - 토큰 분류의 경우 - 각 셀에서 출력 계산
        
        ![2024-12-19_02-48-23.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/1ff9cfe3-1337-4e40-888a-1465c010d218/2024-12-19_02-48-23.jpg)
        

### (2) RNN 훈련 방식

1. BPTT (Backpropagation through time)
    - RNN의 기본 훈련 방식
    - 시간을 역순으로 역전파를 진행하여 수행
    - 특정 입력이 출력에 미치는 영향을 평가하기 위해, 역전파는 특정 계산 패스를 따라 진행
        
        ![2024-12-19_02-50-15.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/d7dac0fd-7989-4ea2-83ee-d629145dd7ba/2024-12-19_02-50-15.jpg)
        
    - 문제점
        - **시간이 깊어질수록** 거쳐야 하는 경로가 길어져, RNN의 훈련이 점점 더 어려워짐 (경사도 소멸)
        - 장기기억 (Long-term memory) 의존성
            - 순차가 길어질 수록 앞쪽 데이터가 뒤쪽으로 잘 전달되지 않음
            - 단기기억(Short-term memory)에만 결과가 의존함

1. LSTM (Long Short Term Memory)
    - RNN의 장기기억 의존성 문제를 해결하기 위해 고안
    - 장단기 기억을 선택적으로 고려할 수 있는 구조(Forget gate)를 포함
    - 은닉정보(ℎ)와 구별되는 셀 상태(𝐶)에 대한 변수를 추가
        
        → 은닉정보는 단기기억을, 셀 상태는 장기기억을 표현
        
        ![2024-12-19_02-51-34.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/def5e2ec-8b79-4070-8c89-45d426acd89f/2024-12-19_02-51-34.jpg)
        
    - 복잡한 구성
        
        ![2024-12-19_02-52-16.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/92f09311-dd67-4649-9454-32fce2dbf8b3/2024-12-19_02-52-16.jpg)
        
        - 현재 셀 상태 ($𝐶_𝑡$)
            - 이전 셀 상태와 임시 셀 상태를 선택적으로 고려해 계산 (장기기억)
        - 은닉 출력 ($ℎ_𝑡$)
            - 셀상태를 선택적으로 고려하여 계산 (단기기억)
            - 새로운 입력값($x_t$)와 이전 은닉 출력값($h_{t-1}$)을 하이퍼탄젠트 계산 → 임시 셀 상태 ( $\bar{𝐶_𝑡}$) 설정
        - 3개의 게이트
            - 각 게이트가 임시 셀 상태 ( $\bar{𝐶_𝑡}$)에 셀 상태 및 은닉 상태정보를 선택적으로 고려해 성능을 향상
            1. 입력 게이트 (Input gate, $i_𝑡$)
                - 임시 셀 상태를 선택적으로 고려하기 위한 입력
            2. 망각 게이트 (Forget gate, $𝑓_𝑡$)
                - 이전 셀 상태($𝐶_{𝑡−1}$)를 고려하는 정도에 대한 계산
            3. 출력 게이트 (Output gate, $𝑜_𝑡$)
                - 은닉정보 출력을 선택적으로 계산하기 위한 모듈

1. GRU (Gate Recurrent Unit)
    - LSTM의 복잡한 구조를 단순화 & 유사한 성능
    - 은닉정보($ℎ$)와 셀 상태($𝐶$)를 통합
    - 업데이트 게이트 ($𝑧_𝑡$) : 입력 게이트와 망각 게이트를 통합
        
        ![2024-12-19_03-03-18.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/f081403d-9e4b-47c9-a54e-613ab8c83a6b/2024-12-19_03-03-18.jpg)
        
    - 계산 방법
        
        ![2024-12-19_03-03-58.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/ccb9fb5a-2fcf-4070-b474-3e6142b4d867/2024-12-19_03-03-58.jpg)
        
        - $z_t$와 $1-z_t$라는 비율로 현재와 과거의 은닉 출력을 혼합하여 결과를 도출

### (3) RNN 확장

- 다계층 RNN
    - 심층 구조로 더 복잡한 패턴의 파악이 가능
- 양방향 RNN
    - 양방향 구조로 전체 정보를 고려하는데 유리
    - ex. 문장 전체를 보고 분류, 빈 칸 채우기
    - 순차/시계열 예측에는 사용이 어려움 (미래 정보를 볼 수 없음)

- 감성분석 (Sentiment Analysis)
    - 리뷰 문장의 긍정/부정 등을 판단하는 문제
        
        ![2024-12-19_03-12-01.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/12facc56-57a0-40b8-9705-d7cb9e997069/2024-12-19_03-12-01.jpg)
        
        1. 문장의 벡터화 → One-hot Encoding으로 고차원 맵핑을 수행
        2. 데이터를 임베딩하여 저차원으로 변환
        3. 변환된 데이터는 RNN에 입력되어 단어 순서대로 정보를 히든 레이어에 전달
        4. 히든 레이어의 최종 결과로 나온 값을 폴리 커넥티드 레이어에 적용

### (3) RNN 구현

- 텍스트 벡터화 (One-hot Encoding)
    
    ```python
    vectorize_layer = tf.keras.layers.TextVectorization(
        max_tokens = 10000,            # number of tokens
        output_sequence_length = 64    # maximum length of sequence
    )
    vectorize_layer.adapt(xtrain)
    ```
    
    ```python
    vectorize_layer('I love you')
    ```
    
    ![2024-12-19_23-43-04.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/9de3bf5b-2157-46d3-97d3-b3980ede551d/2024-12-19_23-43-04.jpg)
    

- 모델링
    
    ```python
    model = tf.keras.Sequential()
    model.add( vectorize_layer )
    model.add( tf.keras.layers.Embedding(10000,32) )  # 10000x32 임베팅 행렬
    model.add( tf.keras.layers.SimpleRNN(16) )        # 은닉정보 벡터크기 = 16
    model.add( tf.keras.layers.Dense(2,activation='softmax') )  # 문장 분류
    
    model.summary()
    ```
    
    ![2024-12-19_23-43-45.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/f1f5220c-cfc2-4014-ad3d-ffc1b8764d47/2024-12-19_23-43-45.jpg)
    

- 모델 훈련
    
    ```python
    model.compile(
        optimizer = 'adam',
        loss = 'sparse_categorical_crossentropy',
        metrics = ['accuracy']
    )
    model.fit(xtrain,ytrain,epochs=5)
    model.predict(['I love this movie', 'Very terrible movie'])
    ```
    
    ![2024-12-19_23-44-28.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/86e31abb-64ab-4275-baa9-bc9761f89ca1/2024-12-19_23-44-28.jpg)
    

- 사전 학습 모델 (Hugging Face) 활용
    
    ```python
    from transformers import pipeline
    
    classifier = pipeline(task='text-classification')
    classifier(['I love this movie', 'Very terrible movie'])
    ```
    
    ![2024-12-19_23-47-34.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/6efb01d1-b992-482e-8aa6-abc7c332cf29/2024-12-19_23-47-34.jpg)
    

## 3. 언어 모델

### (1) Seq2Seq 모델

- 언어 모델 (Language Model)
    - 인간 언어 또는 프로그래밍 언어를 수학적으로 표현
    - 단어(토큰)의 출현과 순서에 대한 확률을 나타내는 모델
        - ex. Pr(나는 학교에 간다) > Pr(나는 학교에 온다) > Pr(나는 간다 학교에)
    - 단어 벡터 활용
        
        ![2024-12-19_14-18-12.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/c82ade55-0c54-4826-a3bb-76cd2f8d8e6c/2024-12-19_14-18-12.jpg)
        
    - 번역, 오타 교정, 빈칸 채우기 등에 활용

- Seq2Seq 모델
    - 순차적인 데이터를 입력으로 받아 출력 역시 순차적인 데이터를 생성하는 모델
    - 문장 분류 뿐만 아니라 번역, Q&A 등에도 사용될 수 있음
    - 인코더 & 디코더로 구성
        - 멀티 레이어 RNN 구조 활용
        - 인코더 : 입력 문장을 이해하는 모듈 (양방향)
        - 디코더 : 출력 문장 (단방향)
        - 출력이 인코더의 은닉정보에서 출발해서 단방향 순차적 진행 → 문장의 길이가 길어질수록 성능 저하
        - ex. 번역 예시
            
            ![2024-12-19_14-19-05.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/0acea99b-3bfb-4d27-a69b-9285aba7e2d3/2024-12-19_14-19-05.jpg)
            

### (2) Attention

- Attention Mechanism
    - 디코더가 새로운 단어를 예측할 때, 인코더 출력 시퀀스의 각 단어 중 어떤 부분에 더 가중치를 부여해야 하는지를 학습하는 방법
    - 특정 단어가 이전 단어를 직접 참조할 수 있는 경로를 제공하여 이전 정보에 대한 접근성을 높임
        - 이전 RNN 노드의 정보를 다시 이용해서 최종 출력을 결정 (인코더의 출력에 의존 X)
        - Attention 모듈이 어떤 정보를 얼마나 참조할지 결정 - 가중치에 따라 인코더 출력값 반영
    - ex. 번역 예시
        
        ![2024-12-19_14-51-09.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/7ea1f1bf-84b5-47a7-907e-e397339f2904/2024-12-19_14-51-09.jpg)
        

- 어텐션의 계산
    
    
    ![2024-12-19_17-59-25.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/3ec34728-4043-4d52-be82-42aa9010d939/2024-12-19_17-59-25.jpg)
    
    - 정보
        - Query(q) : 현재 출력을 계산하려는 셀의 은닉 정보
        - Key(k) : 참조하는 셀의 은닉 정보
        - Value(v) : 참조하는 셀의 값 (보통 Key와 같음)
    
    - Attention(Q, K, V) 계산
        1. Attention Score($e$)
            - 현재 출력하려는 셀의 은닉정보를 query로 사용 → 각 인코더 셀의 키와 내적해 유사도 측정
            - $e_{ij} = q_i^T * k_j$
        2. Distribution($\alpha$)
            - 각 AttentionScore의 총합을 1로 맞추어, 비율로 변환
            - $\alpha_{ij} = \frac{e^{q_i^T k_j}}{\sum_j e^{q_i^T k_j}}$
        3. Attention Output($o$)
            - 각 인코더 셀의 Value와 Distribution의 비중의 곱을 모두 더해 출력값 설정
            - $o_i = \sum_j \alpha_{ij} v_j$
        4. Prediction($y$)
            - Attention Output과 현재 출력하려는 셀의 은닉정보를 FC Layer를 통해 계산
            - $y_i = \text{FC}(q_i, o_i)$

### (3) Self-Attention

- Seq2Seq 인코더의 문제
    - 단어 사이의 거리에 따라 정보가 잘 전달되지 않음
    - 단어 간의 거리가 생기면 연관성을 확인하기 어려움
        
        ![2024-12-19_18-18-22.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/99de2b06-39d1-4aa7-baf0-381b611ee181/2024-12-19_18-18-22.jpg)
        

- Self-Attention
    - 기존 순서 중심의 RNN에서 벗어나, 순서에 얽매이지 않고 모든 단어의 정보를 직접적으로 참조
    - Encoder-decoder 사이의 attention이 아닌, 같은 문장 내에서의 attention
        
        → 각 레이어에서 모든 단어의 정보를 곧바로 얻어오는 것이 가능
        
        ![2024-12-19_18-20-27.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/d5dcc270-48f9-4f10-aff6-f2ea552aff3d/2024-12-19_18-20-27.jpg)
        
    - 구조상 순서 중요 X
        - 단어의 순서가 뒤바뀌어도 모델의 웨이트는 변하지 않음
        
- 위치벡터 (Position Vector)
    - 문장의 순서를 유지하기 위해 단어의 위치를 표현하는 **위치 벡터**를 도입
    - 주기 함수를 이용해 상대적인 위치를 효과적으로 표현
        - 다양한 주기의 sin/cos 함수를 이용하여 표현
        - 장점
            - 절대적인 위치가 중요하지 않고 상대적인 위치가 중요
            - 반복적인 주기함수 → 어느 길이의 문장도 표현이 가능
        - 단점 : 학습해서 얻는 값이 아님 (고정적)
    - 참조 방식
        - 일반적으로 첫 번째 레이어에서만 사용
        - 임베딩된 토큰 벡터 혹은 q, k, v에 더해서 사용
    
- 비선형성의 고려
    - Attention Score는 비선형적으로 결정되지만 Attention Output 자체는 각 단어 정보의 선형적 결합
        - 상위 계층 또한 결국 정보의 선형적 결합이기 때문에 stack을 쌓아도 이득이 없음
    - FFN(Feed Forward)을 삽입하여 해결
        
        ![2024-12-19_18-25-41.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/3fbdeea4-10e1-4743-8647-19b3e97b12cb/2024-12-19_18-25-41.jpg)
        

### (4) 트랜스포머

- 트랜스포머(Transformer)
    - RNN 없이 Attention만으로 인코더-디코더 모델을 구현
    
- 3개의 어텐션 구조
    
    ![2024-12-19_18-28-01.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/05cd419a-758b-4570-ad28-05e34b21711f/2024-12-19_18-28-01.jpg)
    
    1. Encoder Self-Attention
        - 인코더에 사용되는 Self-Attention
        - 입력정보 추출
    2. Encoder-Decoder Cross-Attention
        - 디코더에서 실제로 값을 만들기 위해서 인코더 정보를 참조하기 위한 어텐션 구조
    3. Decoder Masked Self-Attention
        - 디코더에서 사용되는 Self-Attention
        - 출력 문장을 생성할 때 사용되기 때문에 단방향 (역방향은 mask됨)
            
            → 뒤에 나오는 정보들을 학습하지 않음
            
    
- 작동 과정
    1. 입력으로부터 query, key, value를 계산하는 행렬을 별도로 두어 데이터로부터 학습
        - Seq2Seq 모델에서는 RNN 셀의 은닉정보를 이용
    2. Multi-head attention의 사용
        - 한 단어가 여러 개의 단어에 동시에 집중하고 싶은 경우를 고려
        - 하나의 attention이 아니라 여러 개의 attention을 동시에 사용
        - 각각의 attention vector의 크기는 더 작아지기 때문에, 전체 파라미터의 수는 동일
    3. Residual connection
        - 네트워크에서 x와 y간의 차이 (잔차)만을 학습할 수 있도록 설정
        - ResNet, GBM과 잔차 활용 방법 유사
    4. 계층 정규화(Layer Normalization) 사용

### (5) 사전 학습 모델

- 일반 인공지능 모델
    - 원하는 작업에 맞춰 적절한 데이터를 학습
    - 작업에 따라 매번 대규모 훈련이 필요, 작업의 종류에 따라 데이터 확보가 어려움

- 사전 학습 모델 (Pretrained Model)
    - 대규모 데이터(unlabeled)의 일반적인 언어 사이의 관계를 학습
    - 최종적인 모델의 훈련에서 좋은 초기값을 제공
    - 미세학습(Fine-tuning)을 통해 소규모 데이터(labeled)로 원하는 작업을 학습
        
        → 현대의 NLP는 사전학습 + 미세학습으로 이루어짐
        

- 사전 학습 모델의 종류
    1. Encoder들만 : 양방향 → 관계 파악에 유리하지만 훈련의 어려움
    2. Encoder & Decoder : 역할 배분의 어려움
    3. Decoder들만 : 단방향 → 문장 생성은 용이하지만 관계 파악의 어려움
    
    1. Masked Language Model (인코더)
        - 일부 단어를 mask하여 입력을 만들고 해당 단어를 예측하도록 모델을 학습 - 양방향 학습
        - 생성시에는 마지막 단어를 mask 처리
        - ex. BERT
    2. Causal Language Model (디코더)
        - 이전 단어에 기초해 다음 단어를 예측하는 것을 학습 (기존 언어모델과 유사)
        - ex. GPT
            
            ![2024-12-19_23-22-49.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/9e437753-7d0d-40a2-8e7c-a1ba084c3c5a/2024-12-19_23-22-49.jpg)
            
- 학습
    1. 사전 학습 언어모델
        - 언어 자체에 대한 이해를 목적
        - 대규모의 레이블 없는 데이터로 학습, 자기지도학습(self-supervised learning)
        - ex. I love you → I <???> you 를 주고 ???에 들어갈 love 맞추는 문제
    2. 미세학습
        - 특정 작업을 목적 (문장 분류, 번역 등)
        - 소규모의 레이블 있는 데이터로 학습, 지도학습 (supervised learning)
        - ex. l love you → 긍정

- LLM(초거대 언어 모델)
    - 초거대 언어모델의 등장: 더 큰 모델을 더 많은 데이터로 학습
