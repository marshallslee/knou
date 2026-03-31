import numpy as np
import matplotlib.pyplot as plt
import h5py

plt.rcParams['axes.unicode_minus'] = False

# 시드 고정 (매번 결과 똑같이 나오게 하려고 설정함)
np.random.seed(42)

'''
[문제 1] 인공 데이터 생성 및 PCA/LDA 분석
'''

# 1-(1) 데이터 생성 및 산점도 확인
m1, m2 = np.array([0, 0]), np.array([0, 5])
S = np.array([[10, 2], [2, 1]])
n = 100

class1 = np.random.multivariate_normal(m1, S, n)
class2 = np.random.multivariate_normal(m2, S, n)

print("--- [1-(1)] 데이터 생성 완료 ---")
print("클래스1 평균:", class1.mean(0).round(3))
print("클래스2 평균:", class2.mean(0).round(3))

plt.figure(figsize=(7, 5))
plt.scatter(class1[:, 0], class1[:, 1], c='b', alpha=0.5, label='Class 1', s=25)
plt.scatter(class2[:, 0], class2[:, 1], c='r', alpha=0.5, label='Class 2', s=25)
plt.xlim(-10, 10)
plt.ylim(-5, 10)
plt.title('Sample Data')
plt.xlabel('Feature 1'); plt.ylabel('Feature 2')
plt.legend(); plt.grid(ls=':')
plt.savefig('fig1_sample_data.png', dpi=150)
plt.show()


# 1-(2) PCA 및 LDA 고유벡터 계산
X = np.vstack([class1, class2])
X_mean = np.mean(X, axis=0)
X_z = X - X_mean

# PCA: 공분산 행렬 기반 (수식대로 1/N 적용)
cov = (X_z.T @ X_z) / X.shape[0]
evals_p, evecs_p = np.linalg.eigh(cov)
p_idx = np.argsort(evals_p)[::-1]
evals_p = evals_p[p_idx]
evecs_p = evecs_p[:, p_idx]
v_pca = evecs_p[:, 0]

print("\n--- [1-(2)] PCA 결과 ---")
print("고유값:", evals_p)
print("첫번째 주성분 방향:", v_pca)
print("분산 설명 비율: %.1f%%" % (evals_p[0] / np.sum(evals_p) * 100))

# LDA: Within/Between Scatter Matrix 계산
m1_sub, m2_sub = np.mean(class1, 0), np.mean(class2, 0)
Sw = (class1 - m1_sub).T @ (class1 - m1_sub) + (class2 - m2_sub).T @ (class2 - m2_sub)

sb_m1 = (m1_sub - X_mean).reshape(-1, 1)
sb_m2 = (m2_sub - X_mean).reshape(-1, 1)
Sb = n * (sb_m1 @ sb_m1.T) + n * (sb_m2 @ sb_m2.T)

# Sw^-1 * Sb 풀기
l_vals, l_vecs = np.linalg.eig(np.linalg.inv(Sw) @ Sb)
l_sort = np.argsort(np.abs(l_vals))[::-1]
l_vals = l_vals[l_sort]
l_vecs = l_vecs[:, l_sort]
v_lda = l_vecs[:, 0].real

print("\n--- [1-(2)] LDA 결과 ---")
print("고유값:", l_vals)
print("첫번째 판별벡터 방향:", v_lda)

# 벡터 시각화
origin = X_mean
scale = 5

plt.figure(figsize=(7, 5))
plt.scatter(class1[:, 0], class1[:, 1], c='b', alpha=0.3, s=15)
plt.scatter(class2[:, 0], class2[:, 1], c='r', alpha=0.3, s=15)

plt.annotate('', xy=origin + scale * v_pca, xytext=origin,
             arrowprops=dict(arrowstyle='->', color='green', lw=2.5))
plt.annotate('PCA 1st PC', xy=origin + scale * v_pca,
             fontsize=10, color='green', fontweight='bold',
             xytext=(5, 5), textcoords='offset points')

plt.annotate('', xy=origin + scale * v_lda, xytext=origin,
             arrowprops=dict(arrowstyle='->', color='magenta', lw=2.5))
plt.annotate('LDA 1st Vector', xy=origin + scale * v_lda,
             fontsize=10, color='magenta', fontweight='bold',
             xytext=(5, -15), textcoords='offset points')

plt.xlim(-10, 10)
plt.ylim(-5, 10)
plt.title('PCA vs LDA Vector')
plt.grid(ls=':')
plt.savefig('fig2_pca_lda_vectors.png', dpi=150)
plt.show()


# 1-(3) 1차원 투영 히스토그램 비교
proj_p1, proj_p2 = class1 @ v_pca, class2 @ v_pca
proj_l1, proj_l2 = class1 @ v_lda, class2 @ v_lda

fig, ax = plt.subplots(2, 1, figsize=(8, 6))
ax[0].hist(proj_p1, bins=15, alpha=0.5, color='b', label='Class 1')
ax[0].hist(proj_p2, bins=15, alpha=0.5, color='r', label='Class 2')
ax[0].set_title('PCA Projection (1D)')
ax[0].legend(); ax[0].set_xlabel('Projected Value')

ax[1].hist(proj_l1, bins=15, alpha=0.5, color='b', label='Class 1')
ax[1].hist(proj_l2, bins=15, alpha=0.5, color='r', label='Class 2')
ax[1].set_title('LDA Projection (1D)')
ax[1].legend(); ax[1].set_xlabel('Projected Value')

plt.tight_layout()
plt.savefig('fig_projection_comparison.png', dpi=150)
plt.show()


'''
[문제 2] COIL20 데이터셋 분석
'''

# 데이터 로딩 (mat v7.3 버전이라 scipy 말고 h5py로 열어야 됨)
with h5py.File('HW1_COIL20.mat', 'r') as f:
    # 데이터가 (features, samples)로 되어있어서 전치함
    X_raw = np.array(f['X']).T
    Y_raw = np.array(f['Y']).flatten()

print("\n--- [2] COIL20 데이터 로드 완료 ---")
print("데이터 크기:", X_raw.shape, "/ 클래스 수:", len(np.unique(Y_raw)), "개")

# 샘플 이미지 시각화
plt.figure(figsize=(10, 4))
for i in range(10):
    plt.subplot(2, 5, i+1)
    img = X_raw[Y_raw == i+1][0].reshape(32, 32)
    plt.imshow(img, cmap='gray'); plt.axis('off')
    plt.title('Class ' + str(i+1))
plt.suptitle('COIL20 Sample Images (Class 1~10)')
plt.savefig('fig_coil20_samples.png', dpi=150)
plt.show()

# --- PCA 분석 ---
x_c_mean = np.mean(X_raw, axis=0)
x_centered = X_raw - x_c_mean
c_cov = (x_centered.T @ x_centered) / X_raw.shape[0]

evals_c, evecs_c = np.linalg.eigh(c_cov)
s_idx = np.argsort(evals_c)[::-1]
evals_c, evecs_c = evals_c[s_idx], evecs_c[:, s_idx]

# 분산 설명 비율 확인 (95%)
acc_var = np.cumsum(evals_c) / np.sum(evals_c)
d_target = np.argmax(acc_var >= 0.95) + 1

print("\n--- [2] PCA 분석 결과 ---")
print("2차원으로 줄였을때 정보보존율: %.1f%%" % (acc_var[1] * 100))
print("95%% 보존하려면 %d개 주성분 필요 (원래 %d차원)" % (d_target, X_raw.shape[1]))

# 누적 분산 그래프
plt.figure(figsize=(8, 5))
plt.plot(range(1, min(51, len(acc_var)+1)),
         acc_var[:50] * 100, 'bo-', markersize=3)
plt.axhline(y=95, color='r', linestyle='--', label='95%')
plt.axvline(x=d_target, color='g', linestyle='--',
            label=str(d_target) + ' components')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance (%)')
plt.title('PCA Cumulative Explained Variance')
plt.legend(); plt.grid(ls=':')
plt.savefig('fig_cumulative_variance.png', dpi=150)
plt.show()

# PCA 2D 시각화
X_p_2d = x_centered @ evecs_c[:, :2]

classes = np.unique(Y_raw).astype(int)
colors = plt.cm.tab20(np.linspace(0, 1, 20))

plt.figure(figsize=(10, 8))
for i, cls in enumerate(classes):
    mask = (Y_raw == cls)
    plt.scatter(X_p_2d[mask, 0], X_p_2d[mask, 1],
                c=[colors[i]], marker='.', s=50, label='data' + str(cls))
plt.xlabel('1st Principal Component')
plt.ylabel('2nd Principal Component')
plt.title('COIL20 PCA 2-Dim')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
plt.grid(ls=':')
plt.tight_layout()
plt.savefig('fig3_coil20_pca_2d.png', dpi=150, bbox_inches='tight')
plt.show()


# --- LDA 분석 ---
# 고차원 Sw 역행렬 터지는 문제 방지를 위해 PCA로 선축소
X_pca_pre = x_centered @ evecs_c[:, :d_target]

m_total = np.mean(X_pca_pre, 0)
sw_real = np.zeros((d_target, d_target))
sb_real = np.zeros((d_target, d_target))

for c in classes:
    xc = X_pca_pre[Y_raw == c]
    mc = np.mean(xc, 0)
    # Within
    sw_real += (xc - mc).T @ (xc - mc)
    # Between
    mdiff = (mc - m_total).reshape(-1, 1)
    sb_real += len(xc) * (mdiff @ mdiff.T)

# 역행렬 구할때 에러나서 작은 값 더해줌
sw_real += np.eye(d_target) * 1e-6
l_vals_c, l_vecs_c = np.linalg.eig(np.linalg.inv(sw_real) @ sb_real)

l_idx_c = np.argsort(np.abs(l_vals_c.real))[::-1]
l_vals_c = l_vals_c[l_idx_c].real
l_vecs_c = l_vecs_c[:, l_idx_c].real
v_lda_final = l_vecs_c[:, :2]

print("\n--- [2] LDA 결과 ---")
print("상위 5개 고유값:", l_vals_c[:5].round(4))
n_meaningful = np.sum(np.abs(l_vals_c) > 1e-10)
print("의미있는 판별벡터:", n_meaningful, "개 (클래스수-1 = 19개가 최대)")

# LDA 2D 시각화
X_l_2d = X_pca_pre @ v_lda_final

plt.figure(figsize=(10, 8))
for i, cls in enumerate(classes):
    mask = (Y_raw == cls)
    plt.scatter(X_l_2d[mask, 0], X_l_2d[mask, 1],
                c=[colors[i]], marker='.', s=50, label='data' + str(cls))
plt.xlabel('1st LDA Component')
plt.ylabel('2nd LDA Component')
plt.title('COIL20 LDA 2-Dim')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
plt.grid(ls=':')
plt.tight_layout()
plt.savefig('fig3_coil20_lda_2d.png', dpi=150, bbox_inches='tight')
plt.show()

# PCA vs LDA 비교 차트
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

for i, cls in enumerate(classes):
    mask = (Y_raw == cls)
    ax1.scatter(X_p_2d[mask, 0], X_p_2d[mask, 1],
                c=[colors[i]], marker='.', s=50, label='data' + str(cls))
ax1.set_xlabel('1st Principal Component')
ax1.set_ylabel('2nd Principal Component')
ax1.set_title('COIL20 PCA 2-Dim')
ax1.legend(bbox_to_anchor=(1.0, 1), loc='upper left', fontsize=7)
ax1.grid(ls=':')

for i, cls in enumerate(classes):
    mask = (Y_raw == cls)
    ax2.scatter(X_l_2d[mask, 0], X_l_2d[mask, 1],
                c=[colors[i]], marker='.', s=50, label='data' + str(cls))
ax2.set_xlabel('1st LDA Component')
ax2.set_ylabel('2nd LDA Component')
ax2.set_title('COIL20 LDA 2-Dim')
ax2.legend(bbox_to_anchor=(1.0, 1), loc='upper left', fontsize=7)
ax2.grid(ls=':')

plt.suptitle('COIL20: PCA vs LDA', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('fig3_coil20_pca_lda_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n--- 과제 완료 ---")
