# # TensorFlow를 활용한 선형회귀 실습
#
# 이 노트북에서는 TensorFlow를 사용하여 간단한 선형회귀 문제를 풀어봅니다.
#
# ## 주요 단계
# 1. 데이터 생성 (y = 3x + 2 + 잡음)
# 2. 모델 정의 (Dense Layer)
# 3. 손실 함수 및 최적화 방법 설정
# 4. 학습 진행 및 시각화
#

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

# ## 1. 데이터 생성

# y = 3x + 2 + noise
n_samples = 5
X = np.linspace(-5, 5, n_samples)
y = 3 * X + 2 + np.random.normal(0, 1, n_samples)*20

plt.scatter(X, y)
plt.xlabel("X")
plt.ylabel("y")
plt.title("Generated Data")
plt.show()

# ## 2. 모델 정의

model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(shape=(1,)),
    tf.keras.layers.Dense(1)
])

model.summary()

# ## 3. 모델 컴파일
# - 손실 함수: MSE (Mean Squared Error)
# - 최적화 방법: SGD (Stochastic Gradient Descent)

model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
              loss='mse')

# ## 4. 모델 학습

history = model.fit(X, y, epochs=200, verbose=2)

plt.plot(history.history['loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.title('Training Loss')
plt.show()

# ## 5. 결과 확인

W, b = model.layers[0].get_weights()
print(f"학습된 기울기 W: {W[0][0]:.3f}, 절편 b: {b[0]:.3f}")

y_pred = model.predict(X)

plt.scatter(X, y, label='Data')
plt.plot(X, y_pred, color='red', label='Fitted Line')
plt.legend()
plt.show()

# ## 생각해보기
# - 학습 샘플 수가 작아지면 어떻게 되는가?
# 지금 n_samples=5로 되어 있는데 이것도 이미 적은 숫자인 것 같습니다.
# 직선 하나 구하는데 필요한 값은 기울기, 절편 2개뿐인데 점이 5개밖에 없으니까,
# 거기에 맞추다보니 진짜 관계(3x+2)랑 다르게 나온 것 같음.
# 실제로 돌려보니까 W=5.591, b=-6.433 나와서 원래 정답이랑 많이 다르게 나옴.
# 더불어 샘플 개수를 더 줄이면 (2~3개) 더 심하게 틀어질 것으로 판단됨.

# - 학습 샘플에 잡음이 많으면 어떻게 되는가? 이 경우 학습 데이터가 많이 필요한가?
# 코드를 보면 noise에 *20을 곱해서 잡음을 일부러 엄청 크게 만듬.
# 그래서 x값이 같아도 y가 막 튀는거라 모델 입장에서는 어디까지가 진짜 패턴이고,
# 어디까지가 그냥 잡음인지 구별이 잘 안되는듯.
# 이에 데이터를 많이 넣어야 되는게 맞다고 생각이 들며, 데이터가 많아지면 잡음끼리는
# 위아래로 서로 상쇄되면서 평균적으로는 진짜 패턴(3x+2)쪽으로 다시 맞춰질 확률이 높아질 것으로 생각함.
# 따라서, 잡음이 클수록 필요한 데이터양도 같이 늘어나는 느낌?
