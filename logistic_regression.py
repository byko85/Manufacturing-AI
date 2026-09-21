# # TensorFlow를 활용한 로지스틱 회귀 실습
#
# 이 노트북에서는 TensorFlow를 사용하여 간단한 로지스틱 회귀문제를 풀어봅니다.
# 여기서는 iris 데이터셋을 사용하여 꽃의 종류를 분류하는 예제를 다룹니다.
#
# ## 주요 단계
# 1. 데이터 불러오기 및 전처리
# 2. 모델 정의 (Dense Layer)
# 3. 손실 함수 및 최적화 방법 설정
# 4. 학습 진행 및 시각화
#

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
print(tf.__version__)

# ## 1. 데이터 불러오기 및 전처리

# ----------------------
# 1. 데이터 불러오기
# ----------------------
iris = load_iris()
X = iris.data[:, :2]  # 꽃받침 길이, 꽃받침 너비 (2개 feature만 사용해서 시각화 쉽게)
y = (iris.target == 0).astype(np.float32)  # Setosa = 1, Others = 0

print("X shape:", X.shape, "y shape:", y.shape)

# train/test 분리
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 표준화
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ## 2. 모델 정의

model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(shape=(2,)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.summary()

# ## 3. 모델 컴파일
# - 손실 함수: MSE (Mean Squared Error)
# - 최적화 방법: SGD (Stochastic Gradient Descent)

model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.1),
              loss="binary_crossentropy",
              metrics=["accuracy"])

# ## 4. 모델 학습

history = model.fit(X_train, y_train,
                    epochs=100,
                    validation_data=(X_test, y_test),
                    verbose=0)

# ----------------------
# 5. 학습 곡선 확인
# ----------------------
plt.plot(history.history['loss'], label="Train Loss")
plt.plot(history.history['val_loss'], label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Binary Crossentropy Loss")
plt.legend()
plt.title("Training & Validation Loss")
plt.show()

# ## 5. 결과 확인

plt.plot(history.history['accuracy'], label="Train Acc")
plt.plot(history.history['val_accuracy'], label="Val Acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Training & Validation Accuracy")
plt.show()

# ----------------------
# 6. 모델 평가
# ----------------------
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"테스트 정확도: {acc:.3f}")

# ----------------------
# 결정 경계 + 전체 데이터 시각화
# ----------------------
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))

grid = np.c_[xx.ravel(), yy.ravel()]
grid = scaler.transform(grid)
probs = model.predict(grid).reshape(xx.shape)

plt.figure(figsize=(8, 6))

# 결정 경계 (배경 색)
plt.contourf(xx, yy, probs, levels=[0, 0.5, 1], alpha=0.3, cmap="bwr")

# 전체 데이터 산점도 (훈련+테스트)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="bwr", edgecolors="k")

plt.xlabel("Sepal length")
plt.ylabel("Sepal width")
plt.title("Decision Boundary with Data (Logistic Regression)")
plt.show()

# ## 생각해보기
# - 로지스틱 회귀 모델과 선형 회귀 모델의 차이는?
# 선형회귀는 결과값이 그냥 실수 아무 값이나 나올 수 있는데(예측값 자체가 숫자),
# 로지스틱회귀는 그 값을 sigmoid에 한번 더 통과시켜서 0~1 사이로 눌러버린다.
# 그래서 확률처럼 쓸 수 있고 0.5 기준으로 나누면 분류가 됨.
# 코드에서도 Dense(1) 뒤에 activation='sigmoid'가 붙은 거랑 loss가 mse에서 binary_crossentropy로 바뀐게 그 차이다.
# 돌려보니 정확도가 1.000 나왔는데, Setosa는 다른 두 종이랑 워낙 확실하게 떨어져있어서 그런 것 같습니다.

# - Learning Rate(학습률)의 역할은 무엇인가요?
# - 너무 크거나 너무 작은 학습률을 사용하면 어떤 일이 발생하나?
# 학습률은 한번 업데이트할때 얼마나 크게 움직일지 정하는 값이라고 배웠으며, 해당 부분이
# 너무 작으면 조금씩만 움직여서 loss가 내려가긴 하는데 너무 느리게 내려가서 epoch를
# 엄청 많이 돌려야 합니다.
# 반대로 너무 크면 한번에 너무 많이 움직여버려서 최적점을 왔다갔다 지나쳐버리거나 아예 loss가 널뛰기하면서 발산할 수도 있음.
# 근데 해당 건은 로지스틱회귀 lr=0.1 쓰고 선형회귀는 lr=0.01 쓰는데 값이 다른거 보면 문제마다 맞는 학습률이 다 다를것 같음.