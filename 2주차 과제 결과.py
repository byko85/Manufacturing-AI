import os
import urllib.request
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import platform
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ===================================================================
# [Step 0-1] 필수 라이브러리 및 한글 폰트 설정
# ===================================================================
np.random.seed(42)

# 시스템별 한글 폰트 자동 설정 (에러 방지)
system_name = platform.system()
if system_name == 'Windows':
    plt.rc('font', family='Malgun Gothic')
elif system_name == 'Darwin':
    plt.rc('font', family='AppleGothic')
else:
    plt.rc('font', family='NanumGothic')

plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지
print("✅ 라이브러리 및 한글 폰트 설정 완료")


# ===================================================================
# [Step 0-2] CWRU 모터 베어링 실데이터 자동 수집
# ===================================================================
def load_cwru_data():
    urls = {
        'normal': 'https://engineering.case.edu/sites/default/files/97.mat',
        'fault': 'https://engineering.case.edu/sites/default/files/105.mat'
    }
    signals = {}
    print("📥 CWRU 모터 진동 실데이터 다운로드 중...")

    for name, url in urls.items():
        filename = f"cwru_{name}.mat"
        try:
            if not os.path.exists(filename):
                # 403 에러 방지용 헤더 추가
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req) as response, open(filename, 'wb') as out_file:
                    out_file.write(response.read())

            mat = scipy.io.loadmat(filename)
            key = [k for k in mat.keys() if 'DE_time' in k][0]
            signals[name] = mat[key].flatten()
            print(f"  - [{name.upper()}] 신호 로드 완료 ({len(signals[name]):,} 개 샘플)")

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
# [Step 0-3] 모터 진동 파형 시각화 (정상 vs 결함)
# ===================================================================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
fs = 12000
t_plot = np.arange(2400) / fs

ax1.plot(t_plot, raw_signals['normal'][:2400], color='blue', alpha=0.8)
ax1.set_title("🟢 정상 모터 (Normal Bearing) - 진폭이 작고 안정적인 진동 파형", fontsize=12, fontweight='bold')
ax1.set_ylabel("Acceleration (g)")
ax1.grid(True, linestyle=':', alpha=0.6)

ax2.plot(t_plot, raw_signals['fault'][:2400], color='crimson', alpha=0.8)
ax2.set_title("🔴 결함 모터 (Inner Race Fault) - 충격성 고진폭 펄스가 발생함", fontsize=12, fontweight='bold')
ax2.set_ylabel("Acceleration (g)")
ax2.set_xlabel("Time (seconds)")
ax2.grid(True, linestyle=':', alpha=0.6)

plt.suptitle("[CWRU 실데이터] 정상 vs 결함 모터 가속도 시간 영역 파형 비교", fontsize=14, fontweight='bold')
plt.tight_layout()
print("\n📊 1/3: 파형 시각화 창을 닫으면 다음으로 진행됩니다.")
plt.show()


# ===================================================================
# Part 1. 나쁜 특징 (Bad Feature)의 문제점 체감
# ===================================================================
def extract_bad_features(signal, segment_len=2048):
    n_seg = len(signal) // segment_len
    feats = []
    for i in range(n_seg):
        seg = signal[i * segment_len:(i + 1) * segment_len]
        feat_mean = np.mean(seg)
        feat_inst = seg[10]
        feats.append([feat_mean, feat_inst])
    return np.array(feats)


bad_normal = extract_bad_features(raw_signals['normal'][:102400])
bad_fault = extract_bad_features(raw_signals['fault'][:102400])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.hist(bad_normal[:, 0], bins=20, alpha=0.6, color='blue', label='Normal (정상)')
ax1.hist(bad_fault[:, 0], bins=20, alpha=0.6, color='red', label='Fault (결함)')
ax1.set_title("[나쁜 특징 1D] 산술 평균 (Mean) - 정상과 불량 분포가 100% 겹침", fontsize=11, fontweight='bold')
ax1.set_xlabel("Feature Value (Mean)")
ax1.set_ylabel("Count")
ax1.legend()
ax1.grid(True, linestyle=':', alpha=0.6)

ax2.scatter(bad_normal[:, 0], bad_normal[:, 1], c='blue', alpha=0.7, label='Normal (정상)')
ax2.scatter(bad_fault[:, 0], bad_fault[:, 1], c='red', alpha=0.7, label='Fault (결함)')
ax2.set_title("[나쁜 특징 2D] (Mean vs Instant Point) - 데이터가 뒤섞여 구별 불가", fontsize=11, fontweight='bold')
ax2.set_xlabel("Bad Feature 1: Mean")
ax2.set_ylabel("Bad Feature 2: Instant Value")
ax2.legend()
ax2.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
print("📊 2/3: 나쁜 특징 시각화 창을 닫으면 다음으로 진행됩니다.")
plt.show()


# ===================================================================
# Part 2. 좋은 특징 (Good Feature) 추출을 통한 명확한 클래스 분리 체감
# ===================================================================
def extract_good_features(signal, segment_len=2048, fs=12000):
    n_seg = len(signal) // segment_len
    feats = []
    for i in range(n_seg):
        seg = signal[i * segment_len:(i + 1) * segment_len]
        rms = np.sqrt(np.mean(seg ** 2)) * 10.0
        fft_vals = np.abs(np.fft.rfft(seg))
        freqs = np.fft.rfftfreq(segment_len, 1 / fs)
        high_ratio = (np.sum(fft_vals[freqs > 2000] ** 2) / (np.sum(fft_vals ** 2) + 1e-8)) * 10.0
        feats.append([rms, high_ratio])
    return np.array(feats)


good_normal = extract_good_features(raw_signals['normal'][:102400])
good_fault = extract_good_features(raw_signals['fault'][:102400])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.hist(good_normal[:, 0], bins=15, alpha=0.7, color='blue', label='Normal (정상)')
ax1.hist(good_fault[:, 0], bins=15, alpha=0.7, color='red', label='Fault (결함)')
ax1.set_title("[좋은 특징 1D] 진동 RMS - 두 클래스의 분포가 깔끔하게 분리됨!", fontsize=11, fontweight='bold')
ax1.set_xlabel("Feature Value (RMS)")
ax1.set_ylabel("Count")
ax1.legend()
ax1.grid(True, linestyle=':', alpha=0.6)

ax2.scatter(good_normal[:, 0], good_normal[:, 1], c='blue', s=50, alpha=0.8, label='Normal (정상)')
ax2.scatter(good_fault[:, 0], good_fault[:, 1], c='red', s=50, alpha=0.8, label='Fault (결함)')
ax2.set_title("[좋은 특징 2D] (RMS vs High-Freq Energy) - 명확하게 이격된 2개 군집", fontsize=11, fontweight='bold')
ax2.set_xlabel("Good Feature 1: Vibration RMS")
ax2.set_ylabel("Good Feature 2: High-Freq Spectral Energy")
ax2.legend()
ax2.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
print("📊 3/3: 좋은 특징 시각화 창을 닫으면 최종 결과가 출력됩니다.")
plt.show()


# ===================================================================
# Part 3. 특징의 정량적 평가 (Fisher Separability) 및 AI 분류 성능 비교
# ===================================================================
def calculate_fisher_ratio(feat1, feat2):
    mu1, mu2 = np.mean(feat1), np.mean(feat2)
    var1, var2 = np.var(feat1), np.var(feat2)
    return ((mu1 - mu2) ** 2) / (var1 + var2 + 1e-8)


fisher_bad = calculate_fisher_ratio(bad_normal[:, 0], bad_fault[:, 0])
fisher_good = calculate_fisher_ratio(good_normal[:, 0], good_fault[:, 0])

X_bad = np.vstack([bad_normal, bad_fault])
X_good = np.vstack([good_normal, good_fault])
y_all = np.array([0] * len(bad_normal) + [1] * len(bad_fault))

clf_bad = LogisticRegression().fit(X_bad, y_all)
acc_bad = accuracy_score(y_all, clf_bad.predict(X_bad)) * 100

clf_good = LogisticRegression().fit(X_good, y_all)
acc_good = accuracy_score(y_all, clf_good.predict(X_good)) * 100

print(f"\n=============================================")
print(f"📊 [특징 평가 비교 결과]")
print(f"=============================================")
print(f"  1) 나쁜 특징 (Mean)     -> Fisher 분리도 지수: {fisher_bad:.4f}  | AI 분류 정확도: {acc_bad:.1f}%")
print(f"  2) 좋은 특징 (RMS)      -> Fisher 분리도 지수: {fisher_good:.4f} | AI 분류 정확도: {acc_good:.1f}%")


# ==========================================================================
# Part 4. [학생 실습 TODO 해결] 파두율 (Crest Factor) 계산
# ==========================================================================
def extract_crest_factor(signal, segment_len=2048):
    n_seg = len(signal) // segment_len
    crest_factors = []
    for i in range(n_seg):
        seg = signal[i * segment_len:(i + 1) * segment_len]
        rms = np.sqrt(np.mean(seg ** 2))

        # 빈칸 완성 및 주석 해제
        peak_val = np.max(np.abs(seg))
        crest_factor = peak_val / (rms + 1e-8)
        crest_factors.append(crest_factor)

    return np.array(crest_factors)


cf_normal = extract_crest_factor(raw_signals['normal'][:102400])
cf_fault = extract_crest_factor(raw_signals['fault'][:102400])
print(f"\n=============================================")
print(f"✅ 파두율(Crest Factor) 계산 실습 완료!")
print(f"=============================================")
print(f"   정상 평균 파두율: {np.mean(cf_normal):.2f} | 결함 평균 파두율: {np.mean(cf_fault):.2f}")
print(f"=============================================")

"""
==========================================================================
✍️ [질문 1] 좋은 특징의 조건 관찰 서술형 답변
==========================================================================
나쁜 특징(산술 평균)은 정상 데이터와 결함 데이터의 분포가 서로 완벽히 겹쳐서 분류가 불가능하지만, 
좋은 특징(진동 RMS)은 클래스 내 밀집도와 클래스 간 거리를 극대화하여 데이터를 명확하게 분리해 줍니다. 
이렇게 입력 데이터 자체가 명확하게 분리되어야 AI 모델이 쉬운 경계선을 긋고 높은 정확도로 결함을 인식할 수 있기 때문입니다.
"""