## 1. 시각인지 개요

### (1) 영상 인식 작업

- 컴퓨터 비전 (CV)
    - 사람이 이미지를 보고 다양한 정보를 쉽게 파악하는 능력을 컴퓨터도 모방할 수 있도록 연구하는 분야
    - 1차적으로 인간과 동등한 수준의 인식과 이해를 목표

- 영상인식
    - 영상인식을 위해 필요한 기본 작업
        1. 이미지 분류 (Classification): 이미지 전체 주제에 대한 분류
        2. 이미지 분류 + Localization: 해당 분류에 대한 이미지 내의 위치 결정
        3. 객체 탐지 (Object Detection): 다수의 객체에 대한 탐지와 분류
        4. 객체 분할 (Instance Segmentation): 픽셀 단위로 객체를 탐지
            
            ![2024-12-15_03-09-20.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/06da0fdd-7d37-4942-9f7f-5293bfcb0758/2024-12-15_03-09-20.jpg)
            
    - 응용
        - 분류/객체 탐지의 응용: Facial detection, pose detection
        - 상호작용: behavior detection, interaction detection
        - 동영상 적용: object tracking
    - 멀티모달
        - 이미지 + 언어에 대한 인식
        - Captioning, Image Q&A에서 활용

- 시각인지
    - 전통적인 시각인지 방법
        - 시각 인지를 위해 필요한 정보를 “인간이 직접” 특정
        - 알고리즘을 사용해 정보를 추출해 이미지 분류나 객체 검출을 수행하는 방식
            
            ![2024-12-15_03-13-47.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/a4b0e65d-0607-4a4f-b952-2c912630724f/2024-12-15_03-13-47.jpg)
            
    - 딥러닝 기반의 시각인지
        - 필요한 정보를 모델이 자동으로 추출 - 합성곱 신경망(CNN)
        - 대규모 데이터와 높은 계산량이 필요

### (2) 시각인지 구현

- tensorflow 활용
    
    ```python
    import tensorflow as tf
    import numpy as np
    import matplotlib.pyplot as plt
    ```
    

- 데이터셋 불러오기
    
    ```python
    mnist = tf.keras.datasets.mnist
    (xtrain,ytrain), (xtest,ytest) = mnist.load_data()
    print( xtrain.shape, xtest.shape )
    ```
    
    ![2024-12-15_11-18-50.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/b33f0990-8fd3-4bb8-b885-f2758a6134b7/2024-12-15_11-18-50.jpg)
    
    → 총 6만개의 28x28 크기의 흑백 이미지로 훈련 진행
    

- 데이터셋 샘플 시각화로 확인
    
    ```python
    plt.gray()
    plt.matshow( xtrain[0] )
    plt.show()
    ```
    
    ![2024-12-15_11-21-05.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/04becaa7-83bc-4459-a61c-81fb1c07be4b/2024-12-15_11-21-05.jpg)
    

- 이미지 전처리
    
    ```python
    print( xtrain.max(), xtrain.min() )
    ```
    
    > 255 0
    
    ```python
    # 전처리: min-max normalization
    xtrain, xtest = xtrain/255.0, xtest/255.0
    ```
    

## 2. 합성곱 신경망 (CNN)

- 합성곱 신경망 (CNN: Convolutional Neural Network)
    - 이미지 데이터 처리를 위한 가장 기본적인 신경망
    - 지역적 패턴을 찾아 Shift Invariance를 구현 → 패턴의 위치가 변해도 인식 가능하도록 함
    - 시계열 데이터와 같이 지역적 패턴이 중요한 경우 사용되기도 함 - 데이터의 연속적인 패턴 검색
        
        ![2024-12-15_03-18-46.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/0973614f-cdc9-48fb-ad95-de159a74bdad/2024-12-15_03-18-46.jpg)
        

### (1) 이미지 데이터

- 이미지 데이터
    - 기본적으로 픽셀 기반의 이미지를 처리
    - 픽셀 기반의 이미지 : 행렬 형태 - 각 원소는 해당 픽셀의 색상 표현
        1. 흑백 이미지 : 색상을 0에서 255까지의 단계로 나누어, 0은 검은색, 255는 흰색을 표현
            
            ![2024-12-15_03-19-33.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/f801354e-ff93-468d-9e33-514787d870f8/2024-12-15_03-19-33.jpg)
            
        2. 컬러 이미지 : 색의 **삼원색**인 RGB(빨간색, 초록색, 파란색)를 이용한 3차원 행렬 형태로 저장
            
            ![2024-12-15_03-20-33.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/21ab7f49-ff95-4690-bbea-90e1104f990d/2024-12-15_03-20-33.jpg)
            
        - 채널: 이미지를 표현하는 행렬의 수 (흑백 1, 컬러 3)
        - 8비트(0~255) x3 → 24비트 컬러

- 이미지 전처리
    - 현실적인 이미지 전처리는 이미 수행되어 있다고 가정 (노이즈 및 광원 효과 제거, 회전변환 등)
    1. 픽셀값 변환
        - Min-man 정규화로 색상값 0~255에서 0~1로 변환
    2. 이미지 크기 변환
        - 판단 과정에서는 이미지의 화질이 큰 차이 발생하지 않으며, 계산값이 크게 증가함 → 저화질 변환
        - 표준화된 크기로 변환 : MNIST(32x32), ImageNet(256x256)
        - Fitting, padding, cropping 등을 사용
            
            ![2024-12-15_03-43-46.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/a08893bf-c497-418e-8ce4-79b106c65bae/2024-12-15_03-43-46.jpg)
            
    - 전처리된 이미지는 CNN에 투입되어 이미지 분류 작업 수행

- 합성곱 신경망(CNN)의 구조
    1. Convolution Layer와 Pooling Layer가 반복적으로 중첩
        - Convolution Layer : 필터를 통하여 local pattern을 탐지
        - Pooling Layer : 연산량을 줄이기 위해 데이터의 일부만을 탐지
    2. 마지막에 Fully Connected Layer로 분류를 수행
        - Fully Connected Layer : 분류를 위한 일반적인 MLP 구조
            
            ![2024-12-15_03-45-54.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/8e096e50-d8d8-4a23-a5ed-705a6dd36af4/2024-12-15_03-45-54.jpg)
            

### (2) Convolution Layer

- 흑백 - 2차원 텐서의 경우
    - 입력 : 이미지
    - 출력 : 특성 맵 (feature map)
        - 이미지 전체에서 특정 패턴이 얼마나 강하게 나타나는지를 나타내는 결과물
        - 필터가 나타내는 패턴이 이미지의 어느 부분에서 얼마나 강하게 존재하는지 확인
    - 진행 과정
        1. 이미지의 일부분과 커널(필터) 사이의 element-wise 곱을 계산
            - 커널의 값 = 파라미터 → 모델이 데이터 학습을 통해 자동으로 조정
        2. 이미지를 따라 커널을 조금씩 옆으로 옮기며 위 과정을 반복 → **특성 맵** 생성
            
            ![2024-12-15_03-54-22.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/0edd49aa-ac50-4c66-8acc-769ecf1e0d7e/2024-12-15_03-54-22.jpg)
            
    - 추가 요소
        1. Filter size
            - 필터 행렬의 크기
            - ex. 필터 크기 4 → 필터 맵 크기 2x2
        2. Stride
            - 필터가 특성 맵을 생성하기 위해 이미지 위에서 이동하는 간격
            - ex. 스트라이드 2 → 두칸씩 이동 → 필터 맵 크기 2x2
        3. Padding size
            - 원본 이미지를 확장하는 정도
            - 패딩이 없을 경우 패딩 사이즈가 0으로 설정
            - zero padding : 0을 채워 주변을 확장하는 방식
                
                ![2024-12-15_03-57-43.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/2f10a615-7385-48e4-aa6f-4d7fd8472c3b/2024-12-15_03-57-43.jpg)
                
        4. 편향
            - 특성맵에 더해지는 상수 → 비선형성 도입
                
                ![2024-12-15_03-58-43.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/d9d16aec-344b-4fdc-b697-adf8a2404c7e/2024-12-15_03-58-43.jpg)
                
        5. 활성화함수(activation function)
            - 특성 맵에 비선형성을 제공
                
                ![2024-12-15_03-58-54.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/0076516e-6d1b-479f-b35b-699ab414e264/2024-12-15_03-58-54.jpg)
                

- 컬러 - 3차원 텐서의 경우
    - 채널이 여러 개인 경우 필터도 채널의 수 만큼 커져서 합성곱을 수행
        
        ![2024-12-15_03-59-32.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/0858b817-cc18-4e9e-ae61-afdcb5b57919/2024-12-15_03-59-32.jpg)
        
    - 일반적으로 여러 개의 필터를 사용하여 합성곱 수행
        
        ![2024-12-15_04-00-17.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/c698c2b3-e983-4342-a44b-b91c8b3a575e/2024-12-15_04-00-17.jpg)
        

### (3) Pooling Layer

- 특성맵을 다운샘플링하여 맵의 크기를 줄이는 연산
- 일반적으로 2D 맵에 적용
- 채널이 여러 개라도 각 채널별로 풀링을 취하며, 결과적으로 출력 크기는 줄지만 채널의 개수는 변화가 없음
- 학슴해야 할 파라미터가 없어 계산이 단순
1. Max Pooling
    - 최대값을 출력 → 해당 피처가 가장 잘 나타난 부분만을 이용
    - 더 의미 있는 최댓값을 사용하여 특정 패턴을 정확하게 파악할 수 있음
2. Average Pooling
    - 평균값을 출력 → 해당 피처가 평균적으로 나타난 정도를 이용
        
        ![2024-12-15_10-29-53.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/ec7adf06-e7cf-4aba-a71f-833b3ddb6ebe/2024-12-15_10-29-53.jpg)
        

### (4) Fully-Connected Layer

- FC Layer
    - 일반적인 다계층 퍼셉트론(MLP)의 형태로
    - 입력 데이터 : 이미지 정보가 flatten 되어 일렬로 정렬된 형태로 제공
        - Flattening 작업 : 1차원 텐서 변환 - ex. 3x3x64 3D 텐서는 576개의 입력으로 변환
    - 최종적으로 소프트맥스를 사용하여 분류 작업을 수행 → 가능한 클래스의 수에 따라 다양한 분류 노드를 가짐
        
        ![2024-12-15_10-34-06.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/383dae6e-03d8-4252-a819-ba23cb180fe9/2024-12-15_10-34-06.jpg)
        
    
- 전체 CNN 과정 예시
    - Data(28x28) → Conv2D → MaxPooling → Conv2D → MaxPooling → Conv2D → Flatten → FC (ReLU) → FC (Softmax) → Class
        
        ![2024-12-15_10-34-38.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/4ce5c615-8f51-44fa-82e3-eb2b49b936c2/2024-12-15_10-34-38.jpg)
        
        ![2024-12-15_10-35-19.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/6a145ddd-50b7-4430-b0ba-da087dc01229/2024-12-15_10-35-19.jpg)
        
        1.  처음 **흑백** 28x28 이미지로 입력, 컨볼루션 레이어를 거친 후 26x26x32 액티베이션 맵이 생성
        2. 2x2 맥스풀링 과정을 통해 데이터는 13x13x32 **3D 텐서**로 줄어듦
        3. 필터를 3x3x32짜리로 설정하여 컨볼루션과 맥스풀링을 반복 → 최종 출력은 5x5x64 텐서로 변환
        4. 데이터는 576개로 펼쳐진 후, MLP를 통해 최종적으로 10개 카테고리 중 하나로 분류되는 모델이 설계
        5. 최종적으로 약 93,000개의 **파라미터**가 있는 모델이 구현되어, 입력 이미지에서 특정 패턴을 비선형적으로 찾아내는 과정을 거침
            
            

### (5) CNN 모델 구현

- 모델 정의
    
    ```python
    model = tf.keras.models.Sequential()
    model.add( tf.keras.layers.Input(shape=(28,28,1)) )
    model.add( tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu') )
    model.add( tf.keras.layers.MaxPooling2D(pool_size=(2,2)) )
    model.add( tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu') )
    model.add( tf.keras.layers.MaxPooling2D(pool_size=(2,2)) )
    model.add( tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu') )
    model.add( tf.keras.layers.Flatten() )
    model.add( tf.keras.layers.Dense(units=64, activation='relu') )
    model.add( tf.keras.layers.Dense(units=10, activation='softmax') )
    ```
    
- 모델 확인
    
    ```python
    model.summary()
    ```
    
    ![2024-12-15_11-31-36.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/387c78c3-2658-43ed-bcd7-3924679d101b/2024-12-15_11-31-36.jpg)
    

- 모델 훈련 & 평가
    
    ```python
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    model.fit(xtrain,ytrain,epochs=5)
    ```
    
    ![2024-12-15_11-33-06.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/bc05f404-5815-40a2-a0af-33e5c5c676da/2024-12-15_11-33-06.jpg)
    
    ```python
    model.evaluate( xtest, ytest )
    ```
    
    ![2024-12-15_11-33-25.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/221dce28-01bd-4297-a7d9-755f185a5428/2024-12-15_11-33-25.jpg)
    

## 3. 영상처리 모델

### (1) 기반모델

- 기반모델 (Backbone Model)
    - 이미지 분류를 위해 사용되는 사전학습된 대규모 모델
    - 일반적인 CNN 구조를 이용하여 이미지 분류가 가능하나, 실제로 사용하기 위해서는 대규모 모델이 필요
    - 모델을 각 개인이 훈련하여 사용하기는 어렵기 때문에, 대규모로 학습된 공개 모델 활용

1. LeNet
    - 1998년에 발표된 가장 기본적인 **CNN 모델**로, 다양한 시각 인지 문제에 활용
    - 최초의 CNN 모델이지만, 간단한 구조 때문에 딥러닝 모델로는 보통 분류되지 않음

1. AlexNet
    - 2012년 개발된 최초의 딥러닝 기반 이미지 분류 모델
    - ILSVRC 2012에서 top-5 에러율을 16.4%로 낮추며 우승 - 딥러닝의 시작
    - 입력 이미지를 3차원 텐서 형태로 변환 → 맥스 풀링과 컨볼루션 과정을 반복 (중간에 노멀라이제이션 레이어가 포함) → 최종적으로 FC 레이어를 통해 분류
    - 총 8단계의 레이어로 구성 & 약 6천만 개의 파라미터를 가짐
    
2. VGG
    - 2014년 개발됨
    - 7×7 같은 큰 필터 대신 3×3 필터를 많이 쌓아 성능을 올림
        
        → 비슷한 효과, 더 적은 파라메터, 더 큰 비선형적 효과, 학습이 더 힘듦 (컴퓨터 성능 발전으로 가능해짐)
        

1. GoogleLeNet
    - 2014년 개발됨 & ILSVRC 2014 우승
    - 더 깊어진 네트워크 - 22개의 계층
    - 간단한 FC 레이어 구조 & 적은 파라미터 (5백만)로도 좋은 성능을 얻음
    - **인셉션 모듈** 사용
        - 효율적인 계산을 위해 고안된 지역 네트워크
        - 다양한 크기의 필터(ex. 1x1, 3x3 등)를 동시에 사용하여 여러 패턴을 찾아내는 구조
        - conv & pooling을 동시에 수행하여 입력 데이터에서 다양한 패턴을 탐색
            - 이들의 출력을 모두 엮어서 다음 단계의 입력으로 제공
            - 단점 : 필터의 개수가 너무 많아짐
                
                ![2024-12-15_11-10-52.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/34f16e82-5156-4e6a-ad9b-cbdd53fa2916/2024-12-15_11-10-52.jpg)
                
        - 1x1 Convolution 도입
            - 여러 개의 필터를 축약하는 형태
            - 필터의 뎁스를 줄이고, 데이터 크기를 줄이는 데 사용되며 필터를 압축하는 역할
                
                ![2024-12-15_11-11-31.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/f41cf825-4eb9-42a1-a4f8-29aa0241041e/2024-12-15_11-11-31.jpg)
                
            - 전체 구조
                
                ![2024-12-15_11-12-08.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/813d9883-5d83-463c-9d60-e3e03781f491/2024-12-15_11-12-08.jpg)
                
2. ResNet
    - 2015년 마이크로소프트에서 개발 & ILSVRC 2015 우승
    - Residual Block을 통해 매우 깊은 구조(152 계층)를 구현
        - 입력과 출력의 차이만을 학습
        - 차이점의 학습 → 값이 상대적으로 균질, 독립적인 작은 네트워크의 효과
    - ResNet 이후, 이미지 분류 문제는 휴먼 에러보다 낮아짐

1. EfficientNet
    - 모델의 성능은 네트워크의 깊이, 채널 수, 입력 이미지 해상도에 따라 결정됨 →이를 조절하여 성능과 효율을 높여야 함
    - 더 큰 모델은 더 나은 성능을 가져오지만, 이는 계산량과 컴퓨팅 자원 소모의 증가로 이어져 효율적인 모델설계가 필요
    - EfficientNet: 세 가지 요소를 조합하여 AutoML로 최적의 모델을 찾아줌
        
        → 탑1 문제 해결을 위해 정확도를 높이고, 적은 파라미터로 높은 성능을 얻는 것이 목표
        

### (2) VGG 모델 실습

- VGG16 모델 불러오기
    
    ```python
    from tensorflow.keras.applications.vgg16 import VGG16
    from tensorflow.keras.preprocessing import image
    from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
    model = VGG16()
    ```
    

- 이미지 데이터 224x224 크기 조정
    
    ```python
    img = image.load_img('cat.jpeg', target_size=(224,224))
    ```
    

- 이미지 데이터를 3D 텐서 데이터 변환
    
    ```python
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    print(x.shape)
    ```
    
    > (1, 224, 224, 3)
    

- 예측
    
    ```python
    preds = model.predict(x)  # 1,000 개의 클래스에 대한 확률
    print('Predicted:', decode_predictions(preds, top=3)[0])
    ```
    
    ![2024-12-15_11-36-05.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/6f35d10b-ebb2-472d-9867-98ca455698cb/2024-12-15_11-36-05.jpg)
