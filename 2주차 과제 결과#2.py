import os
import platform
import urllib.request
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# ===================================================================
# [Step 0-1] 필수 라이브러리 로드 및 한글 폰트(시스템 자동 감지) 설정
# ===================================================================
np.random.seed(42)

# 파이참 실행 환경에 맞춘 시스템별 한글 폰트 자동 감지
system_name = platform.system()
if system_name == 'Windows':
    plt.rc('font', family='Malgun Gothic')
elif system_name == 'Darwin':
    plt.rc('font', family='AppleGothic')
else:
    plt.rc('font', family='NanumGothic')

plt.rcParams['axes.unicode_minus'] = False
print("✅ 라이브러리 및 한글 폰트 설정 완료")


# ===================================================================
# [Step 0-2] CWRU 실데이터 자동 수집 (서버 오류 방지용 헤더 추가)
# ===================================================================
def load_cwru_data():
    urls = {
        'normal': 'https://engineering.case.edu/sites/default/files/97.mat',
        'fault': 'https://engineering.case.edu/sites/default/files/105.mat'
    }
    signals = {}
    print("📥 CWRU 모터 진동 실데이터 다운로드 중...")

    for name, url in urls.items():
        fn = f"cwru_{name}.mat"
        try:
            if not os.path.exists(fn):
                # 403 Forbidden 에러 방지용 User-Agent 헤더 추가
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req) as response, open(fn, 'wb') as out_file:
                    out_file.write(response.read())

            mat = scipy.io.loadmat(fn)
            key = [k for k in mat.keys() if 'DE_time' in k][0]
            signals[name] = mat[key].flatten()
            print(f"  - [{name.upper()}] 데이터 준비 완료")

        except Exception as e:
            print(f"  - [{name.upper()}] 서버 접근 실패로 대체 모의 신호 생성 (사유: {e})")
            t = np.linspace(0, 10, 120000)
            if name == 'normal':
                signals[name] = 0.05 * np.sin(2 * np.pi * 30 * t) + np.random.normal(0, 0.02, len(t))
            else:
                signals[name] = 0.2 * np.sin(2 * np.pi * 30 * t) + 0.5 * np.sin(2 * np.pi * 150 * t) * (
                            np.sin(2 * np.pi * 5 * t) > 0) + np.random.normal(0, 0.08, len(t))
    return signals


raw_signals = load_cwru_data()


# ===================================================================
# Part 1. 데이터/환경 제약 (Data Constraints)
# ===================================================================
def extract_rms_features(signal, segment_len=2048):
    n_seg = len(signal) // segment_len
    return np.array([np.sqrt(np.mean(signal[i * segment_len:(i + 1) * segment_len] ** 2)) * 10.0 for i in range(n_seg)])


noisy_normal_sig = raw_signals['normal'][:102400] + np.random.normal(0, 0.15, 102400)
noisy_fault_sig = raw_signals['fault'][:102400] + np.random.normal(0, 0.15, 102400)
rms_noisy_norm = extract_rms_features(noisy_normal_sig)
rms_noisy_fault = extract_rms_features(noisy_fault_sig)

clean_norm = raw_signals['normal'][:102400]
clean_fault = raw_signals['fault'][:102400]
rms_clean_norm = extract_rms_features(clean_norm)
rms_clean_fault = extract_rms_features(clean_fault)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.hist(rms_noisy_norm, bins=15, alpha=0.6, color='blue', label='Normal')
ax1.hist(rms_noisy_fault, bins=15, alpha=0.6, color='red', label='Fault')
ax1.set_title("❌ 무제약 데이터: 잡음으로 경계가 흐려짐", fontsize=11, fontweight='bold')
ax1.set_xlabel("RMS Value")
ax1.legend()
ax1.grid(True, linestyle=':', alpha=0.6)

ax2.hist(rms_clean_norm, bins=15, alpha=0.6, color='blue', label='Normal')
ax2.hist(rms_clean_fault, bins=15, alpha=0.6, color='red', label='Fault')
ax2.set_title("⭕ 데이터 제약 부여: 노이즈 제거로 일반화 특징 추출 쉬워짐!", fontsize=11, fontweight='bold')
ax2.set_xlabel("RMS Value")
ax2.legend()
ax2.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
print("\n📊 1/3: 데이터 제약 시각화 창을 닫으면 다음으로 진행됩니다.")
plt.show()

# ===================================================================
# Part 2. 모델 제약 (Model Constraints)
# ===================================================================
X_train_few = np.array([[1.0, 1.2], [1.5, 0.8], [2.0, 1.5], [4.5, 3.2], [5.0, 4.0], [5.2, 3.1]])
y_train_few = np.array([0, 0, 0, 1, 1, 1])

clf_complex = make_pipeline(PolynomialFeatures(degree=4), LogisticRegression(C=1000))
clf_complex.fit(X_train_few, y_train_few)
clf_simple = LogisticRegression(C=0.1).fit(X_train_few, y_train_few)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
xx, yy = np.meshgrid(np.linspace(0, 6, 200), np.linspace(0, 5, 200))

Z1 = clf_complex.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
ax1.contourf(xx, yy, Z1, alpha=0.3, cmap=plt.cm.coolwarm)
ax1.scatter(X_train_few[:, 0], X_train_few[:, 1], c=y_train_few, cmap=plt.cm.bwr, s=80, edgecolors='k')
ax1.set_title("❌ 제약 없는 복잡한 모델: 구불구불 과적합 (일반화 실패)", fontsize=11, fontweight='bold')
ax1.grid(True, linestyle=':', alpha=0.6)

Z2 = clf_simple.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
ax2.contourf(xx, yy, Z2, alpha=0.3, cmap=plt.cm.coolwarm)
ax2.scatter(X_train_few[:, 0], X_train_few[:, 1], c=y_train_few, cmap=plt.cm.bwr, s=80, edgecolors='k')
ax2.set_title("⭕ 단순 모델 제약 부여: 직관적 결정 경계 (우수한 일반화!)", fontsize=11, fontweight='bold')
ax2.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
print("📊 2/3: 모델 제약 시각화 창을 닫으면 다음으로 진행됩니다.")
plt.show()

# ===================================================================
# Part 3. 작업(Task) 제약 (Task Scope Constraints)
# ===================================================================
x_vals = np.linspace(0, 10, 30)
y_continuous = 2.0 * x_vals + np.random.normal(0, 3.0, 30)
y_binary = (y_continuous > 10).astype(int)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.scatter(x_vals, y_continuous, color='purple', alpha=0.7)
ax1.plot(x_vals, 2.0 * x_vals, 'k--', label='True Line')
ax1.set_title("⚠️ 무제약 작업: 연속 실수 예측 (회귀 - 정밀 오차 측정 필요)", fontsize=11, fontweight='bold')
ax1.set_ylabel("Continuous Output Value")
ax1.legend()
ax1.grid(True, linestyle=':', alpha=0.6)

ax2.scatter(x_vals, y_binary, c=y_binary, cmap=plt.cm.bwr, s=60, alpha=0.8)
ax2.set_title("⭕ 작업 제약 부여: 정상(0) vs 불량(1) 2진 분류 (단순 & 직관적)", fontsize=11, fontweight='bold')
ax2.set_ylabel("Class (0: Normal, 1: Fault)")
ax2.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
print("📊 3/3: 작업 제약 시각화 창을 닫으면 최종 결과가 출력됩니다.")
plt.show()


# ==========================================================================
# Part 4. [학생 실습 TODO 해결] 데이터 제약(노이즈 제거 필터) 1줄 적용해 보기
# ==========================================================================
def simple_moving_average_filter(signal, window_size=5):
    return np.convolve(signal, np.ones(window_size) / window_size, mode='same')


# [해답] 제공된 필터 함수를 호출하여 노이즈가 낀 원본 신호에 적용합니다.
filtered_normal = simple_moving_average_filter(noisy_normal_sig)

print(f"\n=============================================")
print("✅ 노이즈 필터링(데이터 제약) 적용 완료!")
print(f"=============================================")
print(f"  - 원본 노이즈 신호 표준편차 : {np.std(noisy_normal_sig):.4f}")
print(f"  - 필터 적용 후 신호 표준편차: {np.std(filtered_normal):.4f}")
print(f"=============================================\n")

"""
==========================================================================
✍️ [질문 1] 제약 조건의 이점 서술 답안
==========================================================================
제조 현장은 데이터가 부족하고 노이즈가 많아 무작정 복잡한 모델을 쓰면 기존 데이터에만 
과도하게 맞춰지는 과적합(Overfitting)이 발생하기 쉽습니다. 반면 데이터 환경을 제약하거나 
문제를 이진 분류 등으로 단순화하면, AI가 학습해야 할 패턴이 뚜렷해져서 새로운 데이터가 
들어와도 높은 예측 정확도(일반화 성능)를 안정적으로 얻을 수 있기 때문입니다.
==========================================================================
"""