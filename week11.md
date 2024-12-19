## 1. 생성모델 개요

- 모델 비교
    
    
    | 구분 | 판별모델 (지도학습)
    (Determinative Model)  | 비지도학습
    (Unsupervised) | 생성모델
    (Generative Model) |
    | --- | --- | --- | --- |
    | 목적 | 주어진 정보(x)로부터 
    관심 정보(y) 결정 | 주어진 정보(x)의 내재된 구조 파악 | 주어진 정보(x)로부터
    유사한 새로운 정보(x') 생성 |
    | 데이터 | (x, y)의 쌍 (정답 레이블 존재) | x만 존재 | x만 존재 |
    | 주요 문제 | x와 y 사이 관계를 나타내는 함수 추론 | x의 확률 분포 혹은 숨겨진 구조 파악 | x의 확률 분포를 파악하고 새로운 샘플 생성 |
    | 예시 | 분류, 회귀 | 군집화, 차원축소 | 이미지 생성, 문장 생성 |

- 언어 모델에도 적용 가능
    - 생성형 모델로 활용 가능: 나는 학교에 <??> → Pr[𝑥 | ’나는’, ’학교에‘]
    - ChatGPT의 경우 : (사용자가 제시하는) 질문 뒤에 나올 문장을 예측하는 구조

- 이미지 생성 모델의 종류
    1. Explicit density
        - 데이터의 확률분포를 학습하여 생성
        - 확률 분포 Pr[X] 가 도출 → 여기에서 임의 추출을 통해 생성
        - VAE
    2. Implicit density
        - 데이터의 확률 분포를 명확히 학습하지 않고 생성
        - Pr[X]가 도출되지 않음, 모델이 내재하여 생성
        - GAN
        
        ![2024-12-20_00-44-59.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/084d28a9-3a7f-4e6e-8e89-cc3716f3e449/2024-12-20_00-44-59.jpg)
        

## 2. VAE

### (1) Autoencoder

- Autoencoder (AE)
    - 비선형적 차원축소를 위한 신경망 모델
    - 기존의 차원축소 : 주성분분석(PCA) → 선형변환이기 때문에 비선형 데이터에는 효과적이지 않음
        - PCA의 학습 : X → (선형압축) → Z → (선형복원) → X’: X와 X’의 차이를 최소화
        - AE의 학습 : X → (비선형압축) → Z → (비선형복원) → X’: X와 X’의 차이를 최소화

- 비선형 압축/복원을 위해 신경망 구조를 사용
    - Encoder: 압축하는 신경망  →  $𝐹(𝑋) = 𝑍$    (Z: 잠재변수)
        - 데이터를 잠재 변수로 압축
    - Decoder: 복원하는 신경망  →  $𝐺 (𝑍) = 𝑋^′$   (X: 관측변수)
        - 이를 복원해 입력 데이터와의 차이를 최소화하여 학습
        - 손실함수: $𝐿 = ||𝐺( 𝐹 (𝑋)) − 𝑋||^ 2$   →    X와 X'의 차이를 최소화
    - 구성
        - 일반적으로 인코더와 디코더는 대칭적으로 구성, MLP/CNN 등이 사용 가능
            
            ![2024-12-20_00-49-40.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/61e4c224-5c10-4148-ad5e-b5996a218698/2024-12-20_00-49-40.jpg)
            

- AE의 변형
    1. Sparse Autoencoder
        - 일부 노드를 제거하여 안정적인 축소 및 복원
        - L1 규제화를 사용하여 구현 가능 → Dropout과 유사하지만 노드가 완전 제거됨
    2. Denoising Autoencoder
        - 노이즈를 넣은 데이터를 입력으로 넣고 원래 데이터를 복원

### (2) Variational Autoencoder

- AE를 사용한 새로운 데이터의 생성?
    
    → 어떤 X를 넣으면 잠재 공간 Z를 거쳐 X랑 유사한 X’가 다시 생성
    
    → 그러나, 잠재 변수(Z)가 어떠한 확률 분포를 갖는지에 대해서는 전혀 관여하지 않음 (알기 어려움)
    
    → 잠재 변수(Z)를 어떠한 특정한 확률 분포를 갖도록 학습 ⇒ 원하는 새로운 이미지 생성 가능
    

- Variational Autoencoder (변이형 오토인코더)
    - 잠재변수가 특정 확률 분포(ex. 표준정규분포)를 갖도록 AE를 구성
    - 특정 확률 분포로부터 샘플을 생성하고, 디코더에 넣어 새로운 데이터 생성
    - 장점
        - 생성모델에 대한 수학적 접근과 확률 근사에 대한 접근법
        - AE에 비해 좀더 밀집된 잠재공간을 갖고 있어 생성에 유리
    - 단점
        - 다른 생성모델에 비해 성능이 떨어짐
    
- 상세 구조
    1. 𝑥는 인코더를 통해 𝑘-차원 평균(𝜇)과 표준편차(𝜎)의 두 잠재변수로 압축됨
    2. $𝑁(𝜇, 𝜎^2)$으로부터 𝑘-차원 잠재변수 𝑧를 임의로 추출
    3. 디코더를 통해 𝑧로부터 𝑥를 다시 재구성
    - ex. k=2
        
        ![2024-12-20_01-15-34.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/573fd808-fa22-413c-aa68-f65ff4a3a6e4/2024-12-20_01-15-34.jpg)
        
    - 손실 함수
        - $𝐿 = ||𝑥 − 𝑑 (𝑧)||^ 2 + 𝐾𝐿 (𝑁 (𝜇, 𝜎^2), 𝑁 (0,1))$
            - $𝑁 (𝜇, 𝜎^2)$ : 실제 z의 분포
            - $𝑁 (0,1))$ : 우리가 원하는 분포
            - $KL$ : 두 분포 사이의 거리
        - 같은 𝑥에 대해서 매번 다른 𝑧가 생성되어 $\hat{𝑥}$을 생성

- 새로운 데이터 생성 : 디코더에 𝑁(0,1) 에서 뽑힌 임의의 값을 넣어 생성
    
    ![2024-12-20_01-21-07.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/eac2472b-bbe6-4039-9460-52eb193865ed/2024-12-20_01-21-07.jpg)
    

### (3) VAE 실습

- 샘플링 함수
    
    ```python
    def sampling(args):
        m, v = args  # v는 log(var)
        epsilon = tf.random.normal( shape=tf.shape(m) )
        return m + tf.math.exp(0.5 * v) * epsilon   # N(0,1) -> N(m,v)
    ```
    

- 인코더 설정
    
    ```python
    input = tf.keras.Input(shape=(input_size,), name='input')
    hidden = tf.keras.layers.Dense(n_hidden, activation='relu', name='hidden')(input)
    z_mean = tf.keras.layers.Dense(latent_size, name='z_mean')(hidden)
    z_var = tf.keras.layers.Dense(latent_size, name='z_var')(hidden)
    z = tf.keras.layers.Lambda(sampling, name='z_sampling')([z_mean, z_var])
    encoder = tf.keras.Model( input, [z_mean, z_var, z] )
    ```
    
    ![2024-12-20_02-17-17.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/40a6945a-83d3-486c-87ee-aaae5ee48c1f/2024-12-20_02-17-17.jpg)
    

- 디코더 설정
    
    ```python
    # 디코더 입력은 latent 변수로부터
    input = tf.keras.Input(shape=(latent_size,), name='input')
    hidden = tf.keras.layers.Dense(n_hidden, activation='relu', name='hidden')(input)
    # 출력은 인코더 입력과 같도록 학습하기 때문에 input_size가 사용
    output = tf.keras.layers.Dense(input_size, activation='sigmoid',name='output')(hidden)
    decoder = tf.keras.Model( input, output, name='decoder' )
    ```
    
    ![2024-12-20_02-18-20.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/78a3e12f-2126-4e73-a42f-2feb5872f955/2024-12-20_02-18-20.jpg)
    

- VAE 모델
    
    ```python
    input = tf.keras.Input(shape=(input_size,), name='input')
    z = encoder(input)[2]  # [0]: z_mean, [1]: z_var, [2]: z
    output = decoder(z)
    model = tf.keras.Model(input, output, name='vae')
    ```
    
    ![2024-12-20_02-18-56.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/208e22d3-2d20-467a-a935-3a5fea985591/2024-12-20_02-18-56.jpg)
    

- 훈련
    
    ```python
    # 훈련 파라메터
    r_loss_factor = 1000
    
    # 손실함수 정의
    def vae_loss(x, recon_x):
      # 재구성을 위한 복원 손실
      m, v, z = encoder(x)
      recon_x = decoder(z)
      r_loss = tf.keras.losses.mse(x,recon_x)
      # 원하는 분포로 맞추기 위한 KL Divergence 손실
      # m = 0, v = log(var) = 0이 되면 최소
      kl_loss = 0.5*tf.reduce_sum(tf.square(m) + tf.exp(v) - v - 1, axis=1)
      return tf.reduce_mean(r_loss*r_loss_factor + kl_loss)
    
    model.compile(optimizer='adam',loss=vae_loss)
    model.fit( xtrain, xtrain, epochs=20 )
    ```
    

- 복원 테스트
    
    ```python
    r_xtest = model.predict(xtest)
    plt.imshow(xtest[0].reshape(28,28), vmin=0, vmax=1, cmap="gray")
    plt.imshow(r_xtest[0].reshape(28,28), vmin=0, vmax=1, cmap="gray")
    ```
    
    ![2024-12-20_02-20-52.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/eb955b47-5289-4349-84ba-c9ccd376cf5d/2024-12-20_02-20-52.jpg)
    

- 생성 테스트
    
    ```python
    # N(0,1)을 따르는 2차원 잠재변수를 입력으로
    new = decoder(np.array([[0,0]]))
    plt.imshow(np.array(new[0]).reshape(28,28), vmin=0, vmax=1, cmap="gray")
    ```
    
    ![2024-12-20_02-21-31.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/ecf1dfc3-79f8-4406-ad0f-f1bff5ee8c65/2024-12-20_02-21-31.jpg)
    

## 3. GAN

### (1) Generative Adversarial Network

- 고차원 데이터의 확률 분포를 찾는 것은 매우 어려운 문제 → 확률 분포를 찾지 않고도 데이터를 생성?

- 게임 이론 (Game Theory)
    - 각 게임 참가자가 자신의 보상을 최대로 하기 위해 노력하는 상황에 대한 분석

- GAN (Generative Adversarial Network, 적대적 생성모델)
    - 2인 게임 상황을 이용하여 새로운 데이터를 생성하는 모델
        - 생성자 (Generator): 실제와 유사한 새로운 데이터를 생성
        - 판별자 (Discriminator): 입력된 데이터가 실제 데이터인지 판별
    - 생성자와 판별자는 적대적 관계로 게임 속에서 서로 성장하는 구조
        
        ![2024-12-20_01-44-21.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/8743292b-e3e1-4790-95a2-848382862fc5/2024-12-20_01-44-21.jpg)
        

- 훈련 과정
    1. 초기 생성자
        - 초기에 생성자는 노이즈로부터 실제와 유사한 데이터를 생성
        - 𝑧: 임의의 노이즈
        - 𝐺(𝑧): 입력 𝑧에 대하여 생성되는 데이터
    2. 초기 판별자
        - 판별자를 이를 실제 데이터와 비교하여 실제인지 가짜인지 판단
        - 𝑥: 실제 데이터
        - 𝐷(𝑥): 입력 𝑥에 대한 판별자의 판단 (0 ~ 1)
    3. 생성자 학습
        - 초기에 생성된 데이터는 구별이 쉽기 때문에 판별자가 쉽게 판단
        - BackPropagation
            - 손실함수 → 생성자가 실제와 유사한 데이터를 생성하도록 학습
            - $𝐿_𝐺 = log (1 − 𝐷( 𝐺 (𝑧)))$
            - 𝐷(𝑥)는 1에 가까울수록, 𝐷(𝐺(𝑧))는 0에 가까울수록 손실이 감소
    4. 판별자 학습
        - 생성자가 잘 훈련되어 실제와 유사한 데이터를 생성하면 판별자가 구별이 어려워져 에러가 증가
        - BackPropagation
            - 손실함수 → 판별자가 더 잘 판별하도록 학습
            - $𝐿_𝐷 = − log 𝐷 (𝑥) − log( 1 − 𝐷 (𝐺( 𝑧)))$
            - 𝐷(𝐺(𝑧)) 는 1에 가까울수록 손실이 감소
    5. 고도화
        - 학습이 잘 된 판별자는 더욱 생성 데이터를 잘 구별 & 생성자는 더욱 실제와 유사한 데이터를 생성
        - 이러한 과정에서 생성자가 최종적으로 진짜에 가까운 데이터를 생성
        - 실제로는 한쪽 손실이 너무 작아지면 다른 쪽과 상관없이 학습이 종료
        
        ![2024-12-20_01-55-50.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/e8f0b1dc-df4d-4c37-9ab7-a6fb3b34b544/2024-12-20_01-55-50.jpg)
        

- 문제점
    1. Oscillation
        - 훈련의 어려움 - 손실이 계속적으로 감소하지 않고 진동
        - 경쟁구조이기 때문에 판별자와 생성자 모두 손실이 낮을 수 없음
    2. Mode collapse
        - 다양성의 부족 - 생성자가 한 가지 유사한 데이터만 생성
        - 분포가 충분히 학습되지 않아 발생

- 훈련을 안정화하기 위한 다양한 기법이 개발
    
    ![2024-12-20_02-00-41.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/edfd69d1-6c01-4d0c-9269-1bae8a4e3915/87cf33c5-ff7d-430c-bdc2-44d1d7465140/2024-12-20_02-00-41.jpg)
    

### (2) GAN의 응용

1. cGAN (Conditional GAN)
    - 특정 조건(레이블)에 해당하는 데이터만 생성
    - ex. 레이블 8(웃는 얼굴)만 생성

1. CycleGAN
    - Unpaired 이미지를 이용하여 style transfer를 구현
    - 하나의 이미지를 다른 형식으로 변형하는 작업
    - 기존 : 일반적으로 paired 이미지가 필요 (ex. 여름-겨울, 사진-고흐풍의그림)

1. SRGAN
    - Super-resolution: 저해상도 → 고해상도
    - 이미지 자체를 조건으로 주어 생성하는 cGAN의 형태
