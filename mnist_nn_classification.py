# # Neural Network Classification: MNIST Dataset
# 
# 이 노트북에서는 **MNIST 손글씨 숫자 (0~9)** 데이터셋을 활용하여 신경망 분류 모델을 학습합니다.
# 
# ## 목표
# 1. 데이터 불러오기 및 전처리
# 2. Neural Network 모델 정의
# 3. 학습 및 학습 곡선 확인
# 4. 성능 평가 및 예측 시각화
# 

# matplotlib 폰트 깨짐 현상 해결
import matplotlib.font_manager as fm
fm.fontManager.ttflist
[f.name for f in fm.fontManager.ttflist]

import matplotlib as mpl
mpl.rcParams['font.family'] = 'NanumGothic'

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist

# ## 1. 데이터 불러오기

(X_train, y_train), (X_test, y_test) = mnist.load_data()
print("원본 데이터 shape:", X_train.shape, y_train.shape)

# ## 2. 데이터 전처리

# 0~255 픽셀값 → 0~1 정규화
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

# (28,28) → (784,) 벡터화
X_train = X_train.reshape(-1, 28*28)
X_test = X_test.reshape(-1, 28*28)

# One-hot 인코딩
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

print("전처리 후 shape:", X_train.shape, y_train.shape)

# ## 3. 모델 정의

model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(shape=(784,)),
    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])

model.summary()

# ## 4. 모델 컴파일

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss="categorical_crossentropy",
              metrics=["accuracy"])

# ## 5. 모델 학습

history = model.fit(X_train, y_train,
                    epochs=10,
                    batch_size=128,
                    validation_split=0.2,
                    verbose=1)

# ## 6. 학습 곡선 확인

plt.plot(history.history['accuracy'], label="Train Acc")
plt.plot(history.history['val_accuracy'], label="Val Acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Training & Validation Accuracy")
plt.show()

plt.plot(history.history['loss'], label="Train Loss")
plt.plot(history.history['val_loss'], label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Training & Validation Loss")
plt.show()

# ## 7. 모델 평가

loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"테스트 정확도: {acc:.3f}")

# ## 8. 예측 예시 시각화

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

idx = 0
plt.figure(figsize=(10,4))
for i in range(10):
    plt.subplot(2,5,i+1)
    plt.imshow(X_test[i+idx].reshape(28,28), cmap="gray")
    plt.title(f"예측:{y_pred_classes[i+idx]}\n실제:{y_true[i+idx]}")
    plt.axis("off")
plt.tight_layout()
plt.show()

failed = y_pred_classes != y_true
failed_idx = np.where(failed)[0]
plt.figure(figsize=(10,4))
for i in range(10):
    plt.subplot(2,5,i+1)
    plt.imshow(X_test[failed_idx[i]].reshape(28,28), cmap="gray")
    plt.title(f"예측:{y_pred_classes[failed_idx[i]]}\n실제:{y_true[failed_idx[i]]}")
    plt.axis("off")
plt.tight_layout()
plt.show()


# ## 생각해보기
# - Training/Validation Graph 에서 과적합(overfitting) 징후가 보이나?
# 실행결과 loss 값은 아래와 같음.
# train loss: 0.3044 -> 0.1154 -> 0.0788 -> 0.0528 -> 0.0402 -> 0.0302 -> 0.0232 -> 0.0192 -> 0.0149 -> 0.0118
# val loss  : 0.1384 -> 0.1113 -> 0.0968 -> 0.0894 -> 0.0872 -> 0.0894 -> 0.0939 -> 0.0884 -> 0.0877 -> 0.0945
#
# train loss는 끝까지 계속 줄어드는데 val loss는 5번째 epoch(0.0872)에서 제일 낮았다가,
# 그 다음부터는 오히려 오르락내리락 하는 걸 보니 과적합 시작되는 것 같습니다.
# accuracy만 보면 val_accuracy도 계속 조금씩 오르긴 해서 (0.9597 -> 0.9759) 헷갈릴수도 있는데
# loss 쪽 보면 확실히 갈라지는게 보임.
# 그럼 epoch를 10보다 적게, 한 5~6정도로 줄이거나 early stopping 쓰면 될 것 같고,
# 아니면 Dropout 같은거 레이어 사이에 넣어주면 과적합이 좀 덜할 것 같다는 생각이 듭니다.
# 이상!!