"""
머신러닝특론 HW1 - Feature Extraction (PCA & LDA)
===================================================
Python을 사용하여 PCA와 LDA를 직접 구현합니다.
sklearn의 PCA, LDA 등 간편 라이브러리는 사용하지 않았습니다.
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py

# 한글 마이너스 기호 깨짐 방지
plt.rcParams['axes.unicode_minus'] = False

# 재현성을 위한 랜덤 시드 고정
np.random.seed(42)


# =====================================================================
# [문제 1] 인공 데이터에 대한 PCA와 LDA
# =====================================================================

# ----- 문제 1-(1): 두 클래스 데이터 생성 및 산점도 (10점) -----

# 파라미터 설정 (과제에서 주어진 값)
mu1 = np.array([0, 0])       # 클래스 1의 평균 벡터
mu2 = np.array([0, 5])       # 클래스 2의 평균 벡터
sigma = np.array([[10, 2],   # 두 클래스 공통 공분산 행렬
                  [2, 1]])
n_samples = 100              # 각 클래스 샘플 수

# 다변량 정규분포에서 데이터 생성
# np.random.multivariate_normal()을 사용하면 평균과 공분산에 맞는 랜덤 데이터를 생성할 수 있다.
# 처음에는 np.random.randn()으로 생성하고 Cholesky 분해를 곱하는 방법도 시도해봤는데,
# multivariate_normal이 더 직관적이고 정확해서 이 방법을 사용했다.
class1_data = np.random.multivariate_normal(mu1, sigma, n_samples)
class2_data = np.random.multivariate_normal(mu2, sigma, n_samples)

print(f"[문제1-(1)] 데이터 생성 완료")
print(f"  Class 1: shape={class1_data.shape}, 샘플 평균={class1_data.mean(axis=0).round(3)}")
print(f"  Class 2: shape={class2_data.shape}, 샘플 평균={class2_data.mean(axis=0).round(3)}")

# 산점도
plt.figure(figsize=(8, 6))
plt.scatter(class1_data[:, 0], class1_data[:, 1],
            c='blue', marker='o', alpha=0.6, edgecolors='k', linewidths=0.5,
            label='Class 1')
plt.scatter(class2_data[:, 0], class2_data[:, 1],
            c='red', marker='o', alpha=0.6, edgecolors='k', linewidths=0.5,
            label='Class 2')
plt.xlim(-10, 10)  # 과제 조건: axis([-10 10 -5 10])
plt.ylim(-5, 10)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Sample Data')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('fig1_sample_data.png', dpi=150)
plt.show()


# ----- 문제 1-(2): PCA와 LDA 벡터 시각화 (20점) -----

# === PCA 직접 구현 ===
# PCA의 핵심: 데이터의 분산이 가장 큰 방향(=공분산 행렬의 고유벡터)을 찾는 것.

# 전체 데이터 합치기
X_all = np.vstack([class1_data, class2_data])  # (200, 2)

# 전체 평균 계산 및 중심화(centering)
# 중심화: 데이터에서 평균을 빼는 것. PCA에서 반드시 필요한 전처리.
mean_all = np.mean(X_all, axis=0)
X_centered = X_all - mean_all

# 공분산 행렬 계산
# C = (1/N) * X_centered^T * X_centered
# 처음에 np.cov()를 사용할까 고민했지만, np.cov()는 1/(N-1)로 나누는 표본 공분산이라
# 수식과 정확히 일치시키기 위해 직접 계산했다.
N = X_all.shape[0]
cov_matrix = (1/N) * X_centered.T @ X_centered

# 고유값, 고유벡터 계산
# eigh(): 대칭행렬(symmetric matrix) 전용 함수. 공분산 행렬은 항상 대칭이므로 적합.
# eigh()는 고유값을 오름차순으로 반환하므로 내림차순 정렬 필요.
eigenvalues_pca, eigenvectors_pca = np.linalg.eigh(cov_matrix)
idx = np.argsort(eigenvalues_pca)[::-1]
eigenvalues_pca = eigenvalues_pca[idx]
eigenvectors_pca = eigenvectors_pca[:, idx]

# PCA 첫 번째 주성분 벡터
pc1 = eigenvectors_pca[:, 0]

print(f"\n[문제1-(2)] PCA 결과")
print(f"  고유값: {eigenvalues_pca}")
print(f"  1st PC 방향: {pc1}")
print(f"  분산 설명 비율: {eigenvalues_pca[0]/np.sum(eigenvalues_pca)*100:.1f}%")

# === LDA 직접 구현 ===
# LDA의 핵심: 클래스 간 분산(Sb)은 최대화, 클래스 내 분산(Sw)은 최소화하는 방향을 찾는 것.
# 수학적으로는 Sw^(-1) * Sb의 고유벡터를 구한다.

# 각 클래스 평균
mean1 = np.mean(class1_data, axis=0)
mean2 = np.mean(class2_data, axis=0)

# Within-class Scatter Matrix (Sw): 클래스 내부의 분산
# Sw = S1 + S2, 여기서 Si = (x - mean_i)^T * (x - mean_i)의 합
diff1 = class1_data - mean1
S1 = diff1.T @ diff1

diff2 = class2_data - mean2
S2 = diff2.T @ diff2

Sw = S1 + S2

# Between-class Scatter Matrix (Sb): 클래스 간의 분산
# Sb = n1*(mean1-mean_total)*(mean1-mean_total)^T + n2*(mean2-mean_total)*(mean2-mean_total)^T
n1, n2 = class1_data.shape[0], class2_data.shape[0]
diff_mean1 = (mean1 - mean_all).reshape(-1, 1)
diff_mean2 = (mean2 - mean_all).reshape(-1, 1)
Sb = n1 * (diff_mean1 @ diff_mean1.T) + n2 * (diff_mean2 @ diff_mean2.T)

# Sw^(-1) * Sb의 고유벡터
Sw_inv = np.linalg.inv(Sw)
eigenvalues_lda, eigenvectors_lda = np.linalg.eig(Sw_inv @ Sb)

# 고유값 내림차순 정렬
idx_lda = np.argsort(np.abs(eigenvalues_lda))[::-1]
eigenvalues_lda = eigenvalues_lda[idx_lda]
eigenvectors_lda = eigenvectors_lda[:, idx_lda]

# LDA 첫 번째 판별 벡터
lda1 = eigenvectors_lda[:, 0].real

print(f"\n[문제1-(2)] LDA 결과")
print(f"  고유값: {eigenvalues_lda}")
print(f"  1st LDA 방향: {lda1}")

# PCA & LDA 벡터를 산점도 위에 시각화
origin = mean_all
scale = 5  # 벡터 길이 스케일 (그래프에서 잘 보이도록)

plt.figure(figsize=(8, 6))

plt.scatter(class1_data[:, 0], class1_data[:, 1],
            c='blue', marker='o', alpha=0.5, edgecolors='k', linewidths=0.3,
            label='Class 1')
plt.scatter(class2_data[:, 0], class2_data[:, 1],
            c='red', marker='o', alpha=0.5, edgecolors='k', linewidths=0.3,
            label='Class 2')

# PCA 벡터 (초록색 화살표)
plt.annotate('', xy=origin + scale * pc1, xytext=origin,
             arrowprops=dict(arrowstyle='->', color='green', lw=2.5))
plt.annotate('PCA 1st PC', xy=origin + scale * pc1,
             fontsize=10, color='green', fontweight='bold',
             xytext=(5, 5), textcoords='offset points')

# LDA 벡터 (마젠타 화살표)
plt.annotate('', xy=origin + scale * lda1, xytext=origin,
             arrowprops=dict(arrowstyle='->', color='magenta', lw=2.5))
plt.annotate('LDA 1st Vector', xy=origin + scale * lda1,
             fontsize=10, color='magenta', fontweight='bold',
             xytext=(5, -15), textcoords='offset points')

plt.xlim(-10, 10)
plt.ylim(-5, 10)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('PCA 1st Principal Component & LDA 1st Vector')
plt.legend(loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('fig2_pca_lda_vectors.png', dpi=150)
plt.show()


# ----- 문제 1-(3): PCA와 LDA 결과 비교 (10점) -----
# 1차원 투영 결과를 히스토그램으로 비교

proj_pca_c1 = class1_data @ pc1
proj_pca_c2 = class2_data @ pc1

proj_lda_c1 = class1_data @ lda1
proj_lda_c2 = class2_data @ lda1

fig, axes = plt.subplots(2, 1, figsize=(10, 6))

axes[0].hist(proj_pca_c1, bins=20, alpha=0.6, color='blue', label='Class 1')
axes[0].hist(proj_pca_c2, bins=20, alpha=0.6, color='red', label='Class 2')
axes[0].set_title('PCA Projection (1D)')
axes[0].legend()
axes[0].set_xlabel('Projected Value')

axes[1].hist(proj_lda_c1, bins=20, alpha=0.6, color='blue', label='Class 1')
axes[1].hist(proj_lda_c2, bins=20, alpha=0.6, color='red', label='Class 2')
axes[1].set_title('LDA Projection (1D)')
axes[1].legend()
axes[1].set_xlabel('Projected Value')

plt.tight_layout()
plt.savefig('fig_projection_comparison.png', dpi=150)
plt.show()


# =====================================================================
# [문제 2] COIL20 데이터에 대한 PCA와 LDA (총 60점)
# =====================================================================

# ----- 데이터 로드 -----
# HW1_COIL20.mat은 MATLAB v7.3 형식이라 scipy.io.loadmat으로는 안 열림.
# 처음에 scipy.io.loadmat()을 사용했다가 에러가 나서, h5py로 바꿨다.
# mat73 라이브러리도 있지만 h5py가 더 범용적이라 이것을 사용.

with h5py.File('HW1_COIL20.mat', 'r') as f:
    # mat 파일은 (features, samples) 형태로 저장됨 → 전치하여 (samples, features)로 변환
    X_train = np.array(f['X']).T   # (280, 1024) - 학습 데이터
    Y_train = np.array(f['Y']).flatten()  # (280,) - 학습 레이블
    # Xt, Yt는 테스트 데이터 → 과제 조건에 따라 사용하지 않음

print(f"\n[문제2] COIL20 데이터 로드 완료")
print(f"  학습 데이터: {X_train.shape} (샘플 수 x 특징 수)")
print(f"  클래스 수: {len(np.unique(Y_train))}, 클래스당 샘플: {X_train.shape[0]//20}")

# 샘플 이미지 확인 (어떤 데이터인지 눈으로 보기 위해)
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i in range(10):
    ax = axes[i // 5, i % 5]
    class_idx = np.where(Y_train == (i + 1))[0][0]
    img = X_train[class_idx].reshape(32, 32)
    ax.imshow(img, cmap='gray')
    ax.set_title(f'Class {i+1}')
    ax.axis('off')
plt.suptitle('COIL20 Sample Images (Class 1~10)', fontsize=14)
plt.tight_layout()
plt.savefig('fig_coil20_samples.png', dpi=150)
plt.show()


# ----- COIL20 PCA 구현 -----

# 평균 계산 및 중심화
mean_coil = np.mean(X_train, axis=0)
X_centered_coil = X_train - mean_coil

# 공분산 행렬 및 고유값 분해
N_coil = X_train.shape[0]
cov_coil = (1/N_coil) * X_centered_coil.T @ X_centered_coil

eigenvalues_coil, eigenvectors_coil = np.linalg.eigh(cov_coil)

# 내림차순 정렬
idx_coil = np.argsort(eigenvalues_coil)[::-1]
eigenvalues_coil = eigenvalues_coil[idx_coil]
eigenvectors_coil = eigenvectors_coil[:, idx_coil]
eigenvalues_coil = np.maximum(eigenvalues_coil, 0)  # 수치 오차로 인한 음수 제거

# 정보보존율(누적 분산 비율) 계산
total_variance = np.sum(eigenvalues_coil)
cumulative_variance = np.cumsum(eigenvalues_coil) / total_variance * 100

# 95%에 해당하는 주성분 수
n_components_95 = np.argmax(cumulative_variance >= 95) + 1
print(f"\n[문제2] PCA 정보보존율 95% → 필요한 주성분 수: {n_components_95}")
print(f"  원래 차원: {X_train.shape[1]} → 축소 차원: {n_components_95}")

# 누적 분산 비율 그래프
plt.figure(figsize=(8, 5))
plt.plot(range(1, min(51, len(cumulative_variance)+1)),
         cumulative_variance[:50], 'bo-', markersize=3)
plt.axhline(y=95, color='r', linestyle='--', label='95% threshold')
plt.axvline(x=n_components_95, color='g', linestyle='--',
            label=f'{n_components_95} components')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance (%)')
plt.title('PCA Cumulative Explained Variance')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('fig_cumulative_variance.png', dpi=150)
plt.show()


# ----- PCA 2차원 투영 -----

W_pca_2d = eigenvectors_coil[:, :2]       # (1024, 2) 투영 행렬
X_pca_2d = X_centered_coil @ W_pca_2d     # (280, 2) 투영된 데이터

classes = np.unique(Y_train).astype(int)
colors = plt.cm.tab20(np.linspace(0, 1, 20))

plt.figure(figsize=(10, 8))
for i, cls in enumerate(classes):
    mask = Y_train == cls
    plt.scatter(X_pca_2d[mask, 0], X_pca_2d[mask, 1],
                c=[colors[i]], marker='.', s=50, label=f'data{cls}')
plt.xlabel('1st Principal Component')
plt.ylabel('2nd Principal Component')
plt.title('COIL20 PCA 2-Dim')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('fig3_coil20_pca_2d.png', dpi=150, bbox_inches='tight')
plt.show()


# ----- COIL20 LDA 구현 (PCA 95% 전처리 후) -----
# 고차원(1024)에서 Sw가 특이행렬이 되는 문제를 해결하기 위해,
# 먼저 PCA로 95% 정보보존 차원까지 축소한 뒤 LDA를 수행한다.

# PCA 95% 차원 축소
W_pca_95 = eigenvectors_coil[:, :n_components_95]
X_pca_95 = X_centered_coil @ W_pca_95  # (280, n_components_95)

n_features = X_pca_95.shape[1]
n_classes = len(classes)
mean_overall = np.mean(X_pca_95, axis=0)

# Within-class & Between-class Scatter Matrix 계산
Sw_coil = np.zeros((n_features, n_features))
Sb_coil = np.zeros((n_features, n_features))

for cls in classes:
    X_cls = X_pca_95[Y_train == cls]
    n_cls = X_cls.shape[0]
    mean_cls = np.mean(X_cls, axis=0)

    # Within-class scatter
    diff = X_cls - mean_cls
    Sw_coil += diff.T @ diff

    # Between-class scatter
    diff_mean = (mean_cls - mean_overall).reshape(-1, 1)
    Sb_coil += n_cls * (diff_mean @ diff_mean.T)

# Sw^(-1) * Sb의 고유벡터 계산
# Sw에 작은 epsilon을 더해 수치적 안정성 확보 (regularization)
# 이 부분은 처음에 에러가 나서 찾아보다 알게 됨
epsilon = 1e-6
Sw_reg = Sw_coil + epsilon * np.eye(n_features)
Sw_inv_Sb = np.linalg.inv(Sw_reg) @ Sb_coil

eigenvalues_lda_coil, eigenvectors_lda_coil = np.linalg.eig(Sw_inv_Sb)

# 내림차순 정렬
idx_lda_coil = np.argsort(np.abs(eigenvalues_lda_coil.real))[::-1]
eigenvalues_lda_coil = eigenvalues_lda_coil[idx_lda_coil].real
eigenvectors_lda_coil = eigenvectors_lda_coil[:, idx_lda_coil].real

print(f"\n[문제2] LDA 결과")
print(f"  상위 5개 고유값: {eigenvalues_lda_coil[:5].round(4)}")
print(f"  유의미한 판별 벡터 수: {np.sum(np.abs(eigenvalues_lda_coil) > 1e-10)} (최대 C-1=19)")

# LDA 2차원 투영
W_lda_2d = eigenvectors_lda_coil[:, :2]
X_lda_2d = X_pca_95 @ W_lda_2d

plt.figure(figsize=(10, 8))
for i, cls in enumerate(classes):
    mask = Y_train == cls
    plt.scatter(X_lda_2d[mask, 0], X_lda_2d[mask, 1],
                c=[colors[i]], marker='.', s=50, label=f'data{cls}')
plt.xlabel('1st LDA Component')
plt.ylabel('2nd LDA Component')
plt.title('COIL20 LDA 2-Dim')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('fig3_coil20_lda_2d.png', dpi=150, bbox_inches='tight')
plt.show()


# ----- PCA와 LDA 나란히 비교 -----

fig, axes = plt.subplots(1, 2, figsize=(18, 7))

for i, cls in enumerate(classes):
    mask = Y_train == cls
    axes[0].scatter(X_pca_2d[mask, 0], X_pca_2d[mask, 1],
                    c=[colors[i]], marker='.', s=50, label=f'data{cls}')
axes[0].set_xlabel('1st Principal Component')
axes[0].set_ylabel('2nd Principal Component')
axes[0].set_title('COIL20 PCA 2-Dim')
axes[0].legend(bbox_to_anchor=(1.0, 1), loc='upper left', fontsize=7)
axes[0].grid(True, alpha=0.3)

for i, cls in enumerate(classes):
    mask = Y_train == cls
    axes[1].scatter(X_lda_2d[mask, 0], X_lda_2d[mask, 1],
                    c=[colors[i]], marker='.', s=50, label=f'data{cls}')
axes[1].set_xlabel('1st LDA Component')
axes[1].set_ylabel('2nd LDA Component')
axes[1].set_title('COIL20 LDA 2-Dim')
axes[1].legend(bbox_to_anchor=(1.0, 1), loc='upper left', fontsize=7)
axes[1].grid(True, alpha=0.3)

plt.suptitle('COIL20 Feature Extraction: PCA vs LDA', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('fig3_coil20_pca_lda_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n===== 모든 과제 수행 완료 =====")
