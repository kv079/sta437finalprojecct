"""
STA437 Final Project — SDSS Stellar Classification Analysis
Central question: How well can photometric measurements alone recover spectroscopic
classifications of celestial objects, and what latent structure underlies these measurements?
"""

# ============================================================
# 0. Imports and Setup
# ============================================================
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from scipy.stats import chi2
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, KernelPCA, FactorAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.covariance import MinCovDet
from sklearn.metrics import adjusted_rand_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

# Consistent style
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.bbox': 'tight',
    'savefig.dpi': 150,
})
COLORS = {'STAR': '#2196F3', 'GALAXY': '#FF5722', 'QSO': '#4CAF50'}
PALETTE = [COLORS['GALAXY'], COLORS['STAR'], COLORS['QSO']]

import os
os.makedirs('/home/user/sta437finalprojecct/figures', exist_ok=True)

# ============================================================
# 1. Data Loading
# ============================================================
df = pd.read_csv('/home/user/sta437finalprojecct/sdss_data.csv')
print(f"Loaded {len(df)} objects, {df.shape[1]} variables")
print(f"Classes: {df['class'].value_counts().to_dict()}")
print(df.describe().round(3).to_string())

# Feature sets
photo_cols   = ['u', 'g', 'r', 'i', 'z']       # photometric magnitudes
spectro_cols = ['redshift']                       # spectroscopic feature
all_phys     = photo_cols + spectro_cols          # all physically meaningful features
meta_cols    = ['run','camcol','field','plate','mjd','fiberid','ra','dec']

# Derived colour indices (standard astronomical practice)
df['u_g'] = df['u'] - df['g']
df['g_r'] = df['g'] - df['r']
df['r_i'] = df['r'] - df['i']
df['i_z'] = df['i'] - df['z']
color_cols = ['u_g', 'g_r', 'r_i', 'i_z']

label_map = {'GALAXY': 0, 'STAR': 1, 'QSO': 2}
y_num = df['class'].map(label_map).values
y     = df['class'].values

# ============================================================
# FIGURE 1 — EDA: Class overview and magnitude distributions
# ============================================================
fig = plt.figure(figsize=(14, 10))
gs  = gridspec.GridSpec(3, 6, figure=fig, hspace=0.45, wspace=0.35)

# Panel A: class counts
ax_pie = fig.add_subplot(gs[0, :2])
counts = df['class'].value_counts()[['GALAXY','STAR','QSO']]
bars = ax_pie.bar(counts.index, counts.values,
                  color=[COLORS[c] for c in counts.index], edgecolor='white', width=0.6)
ax_pie.set_ylabel('Count')
ax_pie.set_title('(A) Class distribution')
for bar, val in zip(bars, counts.values):
    ax_pie.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 60,
                f'{val}\n({val/len(df)*100:.0f}%)', ha='center', va='bottom', fontsize=9)
ax_pie.set_ylim(0, 6200)

# Panel B: redshift by class (log scale)
ax_rs = fig.add_subplot(gs[0, 2:])
for cls in ['STAR','GALAXY','QSO']:
    sub = df[df['class']==cls]['redshift']
    ax_rs.hist(sub.clip(lower=0), bins=60, alpha=0.65, color=COLORS[cls],
               label=cls, density=True)
ax_rs.set_xlabel('Redshift (clipped at 0)')
ax_rs.set_ylabel('Density')
ax_rs.set_title('(B) Redshift distributions by class')
ax_rs.legend()
ax_rs.set_xlim(-0.05, 0.8)

# Panels C–G: magnitude distributions by class
for k, col in enumerate(photo_cols):
    ax = fig.add_subplot(gs[1 + k//3, (k%3)*2:(k%3)*2+2])
    for cls in ['STAR','GALAXY','QSO']:
        sub = df[df['class']==cls][col]
        sub = sub[(sub > 10) & (sub < 26)]   # clip obvious outliers for display
        ax.hist(sub, bins=40, alpha=0.6, color=COLORS[cls], label=cls, density=True)
    ax.set_xlabel(f'{col} (mag)')
    ax.set_ylabel('Density')
    ax.set_title(f'({chr(67+k)}) {col} band')
    if k == 0:
        ax.legend(fontsize=8)

fig.suptitle('Figure 1. SDSS Dataset Overview: Class Distribution and Photometric Magnitudes',
             fontsize=12, fontweight='bold', y=1.01)
plt.savefig('/home/user/sta437finalprojecct/figures/fig1_eda_overview.png')
plt.close()
print("Figure 1 saved.")

# ============================================================
# FIGURE 2 — Correlation matrix and colour-index scatter
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# Left: correlation heatmap of physical features
feat_df = df[all_phys + color_cols].copy()
feat_df.columns = ['u','g','r','i','z','z-shift','u–g','g–r','r–i','i–z']
corr = feat_df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
sns.heatmap(corr, ax=axes[0], annot=True, fmt='.2f', cmap='RdBu_r',
            vmin=-1, vmax=1, linewidths=0.5, annot_kws={'size':8})
axes[0].set_title('(A) Feature correlation matrix\n(photometric bands + redshift + colour indices)')

# Right: g–r vs u–g colour-colour diagram
np.random.seed(42)
idx = np.random.choice(len(df), 3000, replace=False)
sub = df.iloc[idx]
for cls in ['STAR','GALAXY','QSO']:
    mask2 = sub['class'] == cls
    axes[1].scatter(sub.loc[mask2,'u_g'], sub.loc[mask2,'g_r'],
                    s=8, alpha=0.5, color=COLORS[cls], label=cls)
axes[1].set_xlabel('u – g (mag)')
axes[1].set_ylabel('g – r (mag)')
axes[1].set_xlim(-1, 5)
axes[1].set_ylim(-1, 3)
axes[1].legend(markerscale=3)
axes[1].set_title('(B) Colour–colour diagram (g–r vs u–g)\n3,000 random objects')

fig.suptitle('Figure 2. Feature Correlations and Colour-Colour Diagram',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('/home/user/sta437finalprojecct/figures/fig2_correlations.png')
plt.close()
print("Figure 2 saved.")

# ============================================================
# FIGURE 3 — MVN Assessment (Mahalanobis distances + QQ)
# ============================================================
# Use photometric features + redshift, standardised
X_full = df[all_phys].values
scaler = StandardScaler()
X_std  = scaler.fit_transform(X_full)

# --- Mardia's test (manual) ---
def mardia_test(X):
    n, p = X.shape
    mu  = X.mean(axis=0)
    S   = np.cov(X, rowvar=False)
    Sinv = np.linalg.pinv(S)
    diff = X - mu
    D2   = (diff @ Sinv * diff).sum(axis=1)     # Mahalanobis^2
    # Skewness
    b1p = ((diff @ Sinv @ diff.T)**3).mean()
    k_skew = n * b1p / 6
    p_skew = 1 - chi2.cdf(k_skew, df=p*(p+1)*(p+2)/6)
    # Kurtosis
    b2p   = (D2**2).mean()
    mu_k  = p*(p+2)
    sigma_k = np.sqrt(8*p*(p+2)/n)
    z_kurt  = (b2p - mu_k) / sigma_k
    p_kurt  = 2*(1 - stats.norm.cdf(abs(z_kurt)))
    return D2, k_skew, p_skew, b2p, z_kurt, p_kurt

print("\n--- Mardia's test on full photometric+redshift data ---")
D2_all, ks, ps, b2, zk, pk = mardia_test(X_std)
print(f"  Skewness statistic: {ks:.2f},  p = {ps:.4e}")
print(f"  Kurtosis z-score:   {zk:.2f},  p = {pk:.4e}")
print("  => Full sample strongly non-MVN (mixed classes)")

# Per-class MVN tests
print("\n--- Per-class Mardia's test ---")
for cls in ['GALAXY','STAR','QSO']:
    Xc = X_std[y == cls]
    D2c, ks_c, ps_c, b2_c, zk_c, pk_c = mardia_test(Xc)
    print(f"  {cls:8s}  skew p={ps_c:.3e}  kurt p={pk_c:.3e}")

# Figure: Mahalanobis QQ plots for full data and per class (photo only)
X_photo = df[photo_cols].values
X_photo_std = StandardScaler().fit_transform(X_photo)
p = X_photo_std.shape[1]
chi2_quantiles = chi2.ppf(np.linspace(0.005, 0.995, len(X_photo_std)), df=p)

fig, axes = plt.subplots(1, 4, figsize=(15, 4))

# Full data
D2_photo, *_ = mardia_test(X_photo_std)
sorted_D2 = np.sort(D2_photo)
axes[0].plot(np.sort(chi2_quantiles), sorted_D2, '.', markersize=2, alpha=0.4, color='steelblue')
axes[0].plot([0, chi2_quantiles.max()], [0, chi2_quantiles.max()], 'r--', lw=1.5)
axes[0].set_title('All objects (n=10,000)')
axes[0].set_xlabel('χ²(5) quantiles')
axes[0].set_ylabel('Mahalanobis D²')

# Per class
for k, cls in enumerate(['GALAXY','STAR','QSO']):
    Xc  = X_photo_std[y == cls]
    nc  = len(Xc)
    D2c, *_ = mardia_test(Xc)
    chi2q = chi2.ppf(np.linspace(0.005, 0.995, nc), df=p)
    axes[k+1].plot(np.sort(chi2q), np.sort(D2c), '.', markersize=2,
                   alpha=0.4, color=COLORS[cls])
    axes[k+1].plot([0, chi2q.max()], [0, chi2q.max()], 'r--', lw=1.5)
    axes[k+1].set_title(f'{cls} (n={nc})')
    axes[k+1].set_xlabel('χ²(5) quantiles')

fig.suptitle('Figure 3. Mahalanobis Distance Q–Q Plots (photometric features)\n'
             'Deviation from the diagonal indicates departure from multivariate normality.',
             fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('/home/user/sta437finalprojecct/figures/fig3_mvn_qqplot.png')
plt.close()
print("Figure 3 saved.")

# ============================================================
# FIGURE 4 — PCA
# ============================================================
# Use photometric + redshift, standardised
X_std_full = StandardScaler().fit_transform(df[all_phys].values)
pca = PCA()
scores = pca.fit_transform(X_std_full)

print("\n--- PCA explained variance ---")
for i, ev in enumerate(pca.explained_variance_ratio_):
    print(f"  PC{i+1}: {ev:.3f}  (cumulative: {pca.explained_variance_ratio_[:i+1].sum():.3f})")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Scree plot
ax = axes[0]
cumvar = np.cumsum(pca.explained_variance_ratio_)
ax.bar(range(1, len(pca.explained_variance_ratio_)+1),
       pca.explained_variance_ratio_*100, color='steelblue', alpha=0.8)
ax.plot(range(1, len(cumvar)+1), cumvar*100, 'o-', color='darkorange', lw=2, label='Cumulative')
ax.axhline(90, ls='--', color='grey', lw=1, alpha=0.7, label='90% threshold')
ax.set_xlabel('Principal component')
ax.set_ylabel('Explained variance (%)')
ax.set_title('(A) Scree plot')
ax.legend()
ax.set_xticks(range(1, 7))

# PC1 vs PC2 (sample 3000)
ax = axes[1]
np.random.seed(0)
idx2 = np.random.choice(len(scores), 3000, replace=False)
for cls in ['GALAXY','STAR','QSO']:
    m = y[idx2] == cls
    ax.scatter(scores[idx2][m, 0], scores[idx2][m, 1],
               s=8, alpha=0.5, color=COLORS[cls], label=cls)
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
ax.set_title('(B) PC1 vs PC2 (n=3,000)')
ax.legend(markerscale=3)

# Loadings heatmap
ax = axes[2]
loadings = pd.DataFrame(pca.components_.T,
                        index=all_phys,
                        columns=[f'PC{i+1}' for i in range(len(all_phys))])
sns.heatmap(loadings, ax=ax, annot=True, fmt='.2f', cmap='RdBu_r',
            vmin=-1, vmax=1, linewidths=0.5, annot_kws={'size':9})
ax.set_title('(C) PCA loadings')

fig.suptitle('Figure 4. Principal Component Analysis (photometric magnitudes + redshift)',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('/home/user/sta437finalprojecct/figures/fig4_pca.png')
plt.close()
print("Figure 4 saved.")

# ============================================================
# FIGURE 5 — Factor Analysis
# ============================================================
X_photo_std2 = StandardScaler().fit_transform(df[photo_cols].values)

# Determine number of factors via BIC-like criterion (log-likelihood)
bic_scores = []
n_factors_range = range(1, 5)
for nf in n_factors_range:
    fa = FactorAnalysis(n_components=nf, random_state=42, max_iter=1000)
    fa.fit(X_photo_std2)
    bic_scores.append(fa.score(X_photo_std2))

# Fit 2-factor model
fa2 = FactorAnalysis(n_components=2, rotation='varimax', random_state=42, max_iter=1000)
fa2.fit(X_photo_std2)
fa_scores = fa2.transform(X_photo_std2)
loadings_fa = pd.DataFrame(fa2.components_.T,
                            index=photo_cols,
                            columns=['Factor 1', 'Factor 2'])

print("\n--- Factor Analysis (2 factors, varimax) loadings ---")
print(loadings_fa.round(3))

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Log-likelihood by n_factors
ax = axes[0]
ax.plot(list(n_factors_range), bic_scores, 'o-', color='steelblue', lw=2)
ax.set_xlabel('Number of factors')
ax.set_ylabel('Log-likelihood (per sample)')
ax.set_title('(A) Factor model fit vs. number of factors')
ax.set_xticks(list(n_factors_range))

# Loadings heatmap
ax = axes[1]
sns.heatmap(loadings_fa, ax=ax, annot=True, fmt='.2f', cmap='RdBu_r',
            vmin=-1, vmax=1, linewidths=0.5, annot_kws={'size':11})
ax.set_title('(B) Factor loadings (varimax)\n2-factor model')

# Factor scores scatter by class
ax = axes[2]
np.random.seed(1)
idx3 = np.random.choice(len(fa_scores), 3000, replace=False)
for cls in ['GALAXY','STAR','QSO']:
    m = y[idx3] == cls
    ax.scatter(fa_scores[idx3][m, 0], fa_scores[idx3][m, 1],
               s=8, alpha=0.5, color=COLORS[cls], label=cls)
ax.set_xlabel('Factor 1 score')
ax.set_ylabel('Factor 2 score')
ax.set_title('(C) Factor scores by class (n=3,000)')
ax.legend(markerscale=3)

fig.suptitle('Figure 5. Factor Analysis of Photometric Magnitudes (varimax rotation)',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('/home/user/sta437finalprojecct/figures/fig5_factor_analysis.png')
plt.close()
print("Figure 5 saved.")

# ============================================================
# FIGURE 6 — Clustering: GMM vs k-means vs true labels
# ============================================================
# Cluster in PCA(photo only) space — 2 components explain >90%
X_photo_std3 = StandardScaler().fit_transform(df[photo_cols].values)
pca_photo = PCA(n_components=2)
Z_photo   = pca_photo.fit_transform(X_photo_std3)

# With redshift (photo + redshift)
X_full_std = StandardScaler().fit_transform(df[all_phys].values)
pca_full   = PCA(n_components=2)
Z_full     = pca_full.fit_transform(X_full_std)

# Fit GMM (3 components) — photo only and photo+redshift
gmm_photo = GaussianMixture(n_components=3, covariance_type='full', n_init=5,
                             random_state=42).fit(X_photo_std3)
gmm_full  = GaussianMixture(n_components=3, covariance_type='full', n_init=5,
                              random_state=42).fit(X_full_std)
labels_gmm_photo = gmm_photo.predict(X_photo_std3)
labels_gmm_full  = gmm_full.predict(X_full_std)

# Align cluster labels to best matching true label
def align_labels(pred, true_num):
    from itertools import permutations
    best_perm, best_ari = None, -1
    for perm in permutations([0,1,2]):
        remapped = np.vectorize(dict(zip([0,1,2], perm)).get)(pred)
        ari = adjusted_rand_score(true_num, remapped)
        if ari > best_ari:
            best_ari = ari
            best_perm = perm
    return np.vectorize(dict(zip([0,1,2], best_perm)).get)(pred), best_ari

labels_aligned_photo, ari_photo = align_labels(labels_gmm_photo, y_num)
labels_aligned_full,  ari_full  = align_labels(labels_gmm_full,  y_num)
print(f"\nGMM ARI — photo only: {ari_photo:.3f}  |  photo+redshift: {ari_full:.3f}")

cls_names = ['GALAXY','STAR','QSO']
cls_num_map = {0:'GALAXY', 1:'STAR', 2:'QSO'}

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
np.random.seed(2)
idx4 = np.random.choice(len(Z_photo), 3000, replace=False)

# True labels in photo PCA space
ax = axes[0]
for ci, cls in enumerate(cls_names):
    m = y[idx4] == cls
    ax.scatter(Z_photo[idx4][m, 0], Z_photo[idx4][m, 1],
               s=8, alpha=0.4, color=COLORS[cls], label=cls)
ax.set_title('(A) True labels (photo PCA space)')
ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
ax.legend(markerscale=3)

# GMM — photo only
ax = axes[1]
for ci in range(3):
    m = labels_aligned_photo[idx4] == ci
    ax.scatter(Z_photo[idx4][m, 0], Z_photo[idx4][m, 1],
               s=8, alpha=0.4, color=PALETTE[ci], label=cls_names[ci])
ax.set_title(f'(B) GMM clusters — photo only\nARI = {ari_photo:.3f}')
ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
ax.legend(markerscale=3)

# GMM — photo + redshift
ax = axes[2]
for ci in range(3):
    m = labels_aligned_full[idx4] == ci
    ax.scatter(Z_full[idx4][m, 0], Z_full[idx4][m, 1],
               s=8, alpha=0.4, color=PALETTE[ci], label=cls_names[ci])
ax.set_title(f'(C) GMM clusters — photo + redshift\nARI = {ari_full:.3f}')
ax.set_xlabel('PC1 (with redshift)'); ax.set_ylabel('PC2 (with redshift)')
ax.legend(markerscale=3)

fig.suptitle('Figure 6. GMM Clustering: Photometry Alone vs. Photometry + Redshift\n'
             '(ARI = adjusted Rand index; higher = better agreement with true labels)',
             fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('/home/user/sta437finalprojecct/figures/fig6_gmm_clustering.png')
plt.close()
print("Figure 6 saved.")

# ============================================================
# FIGURE 7 — LDA + Cross-validation
# ============================================================
# Features: photo only vs photo + redshift
scaler_photo = StandardScaler()
Xp = scaler_photo.fit_transform(df[photo_cols].values)
scaler_all   = StandardScaler()
Xa = scaler_all.fit_transform(df[all_phys].values)

lda_p = LinearDiscriminantAnalysis()
lda_a = LinearDiscriminantAnalysis()
lda_p.fit(Xp, y)
lda_a.fit(Xa, y)

Z_lda_p = lda_p.transform(Xp)
Z_lda_a = lda_a.transform(Xa)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

acc_lda_p  = cross_val_score(LinearDiscriminantAnalysis(), Xp, y, cv=cv, scoring='accuracy')
acc_lda_a  = cross_val_score(LinearDiscriminantAnalysis(), Xa, y, cv=cv, scoring='accuracy')
acc_lr_p   = cross_val_score(LogisticRegression(max_iter=500, C=1.0, random_state=42),
                              Xp, y, cv=cv, scoring='accuracy')
acc_lr_a   = cross_val_score(LogisticRegression(max_iter=500, C=1.0, random_state=42),
                              Xa, y, cv=cv, scoring='accuracy')

print(f"\n--- 5-fold CV accuracy ---")
print(f"  LDA (photo only):       {acc_lda_p.mean():.3f} ± {acc_lda_p.std():.3f}")
print(f"  LDA (photo+redshift):   {acc_lda_a.mean():.3f} ± {acc_lda_a.std():.3f}")
print(f"  LR  (photo only):       {acc_lr_p.mean():.3f}  ± {acc_lr_p.std():.3f}")
print(f"  LR  (photo+redshift):   {acc_lr_a.mean():.3f}  ± {acc_lr_a.std():.3f}")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
np.random.seed(3)
idx5 = np.random.choice(len(Z_lda_a), 3000, replace=False)

# LDA discriminant plot (photo only)
ax = axes[0]
for cls in cls_names:
    m = y[idx5] == cls
    ax.scatter(Z_lda_p[idx5][m, 0], Z_lda_p[idx5][m, 1],
               s=8, alpha=0.45, color=COLORS[cls], label=cls)
ax.set_xlabel('LD1'); ax.set_ylabel('LD2')
ax.set_title(f'(A) LDA — photo only\n5-fold CV acc = {acc_lda_p.mean():.3f}')
ax.legend(markerscale=3)

# LDA discriminant plot (photo + redshift)
ax = axes[1]
for cls in cls_names:
    m = y[idx5] == cls
    ax.scatter(Z_lda_a[idx5][m, 0], Z_lda_a[idx5][m, 1],
               s=8, alpha=0.45, color=COLORS[cls], label=cls)
ax.set_xlabel('LD1'); ax.set_ylabel('LD2')
ax.set_title(f'(B) LDA — photo + redshift\n5-fold CV acc = {acc_lda_a.mean():.3f}')
ax.legend(markerscale=3)

# CV accuracy comparison bar chart
ax = axes[2]
methods = ['LDA\n(photo)', 'LDA\n(photo+z)', 'LR\n(photo)', 'LR\n(photo+z)']
means   = [acc_lda_p.mean(), acc_lda_a.mean(), acc_lr_p.mean(), acc_lr_a.mean()]
stds    = [acc_lda_p.std(),  acc_lda_a.std(),  acc_lr_p.std(),  acc_lr_a.std()]
clrs    = ['#90CAF9','#1565C0','#FFAB91','#BF360C']
bars    = ax.bar(methods, means, yerr=stds, color=clrs, capsize=5, edgecolor='white', width=0.5)
ax.set_ylim(0.5, 1.05)
ax.set_ylabel('5-fold CV accuracy')
ax.set_title('(C) Classification accuracy\n(error bars = ±1 SD across folds)')
ax.axhline(0.5, ls='--', color='grey', lw=1, alpha=0.5, label='50%')
ax.axhline(1.0, ls='--', color='grey', lw=1, alpha=0.5)
for bar, mean in zip(bars, means):
    ax.text(bar.get_x() + bar.get_width()/2, mean + 0.01, f'{mean:.3f}',
            ha='center', va='bottom', fontsize=9)

fig.suptitle('Figure 7. Linear Discriminant Analysis and 5-fold Cross-Validated Classification Accuracy',
             fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('/home/user/sta437finalprojecct/figures/fig7_lda_cv.png')
plt.close()
print("Figure 7 saved.")

# ============================================================
# FIGURE 8 — Robust PCA + Kernel PCA
# ============================================================
# Robust PCA via MCD
print("\n--- Robust PCA (MCD covariance) ---")
Xp_std = StandardScaler().fit_transform(df[photo_cols].values)

mcd = MinCovDet(support_fraction=0.85, random_state=42)
mcd.fit(Xp_std)

# Mahalanobis distances (robust vs classical)
mu_classic = Xp_std.mean(axis=0)
S_classic  = np.cov(Xp_std, rowvar=False)
diff_c     = Xp_std - mu_classic
maha_classic = np.sqrt(np.einsum('ij,jk,ik->i', diff_c, np.linalg.pinv(S_classic), diff_c))

mu_robust = mcd.location_
S_robust  = mcd.covariance_
diff_r    = Xp_std - mu_robust
maha_robust = np.sqrt(np.einsum('ij,jk,ik->i', diff_r, np.linalg.pinv(S_robust), diff_r))

# Robust outliers threshold: chi2(p, 0.975)
thresh_robust  = np.sqrt(chi2.ppf(0.975, df=len(photo_cols)))
outlier_mask   = maha_robust > thresh_robust
n_outliers     = outlier_mask.sum()
print(f"  Robust Mahalanobis outliers (p=5, α=0.025): {n_outliers} ({n_outliers/len(Xp_std)*100:.1f}%)")
print("  Outlier class breakdown:", pd.Series(y[outlier_mask]).value_counts().to_dict())

# Robust PCA: project onto principal directions of MCD covariance
eigvals, eigvecs = np.linalg.eigh(S_robust)
order = np.argsort(eigvals)[::-1]
eigvecs = eigvecs[:, order]
eigvals = eigvals[order]
Z_robust_pca = (Xp_std - mu_robust) @ eigvecs[:, :2]

# Standard PCA for comparison
Z_std_pca = PCA(n_components=2).fit_transform(Xp_std)

# Kernel PCA (RBF)
kpca = KernelPCA(n_components=2, kernel='rbf', gamma=0.5, random_state=42)
Z_kpca = kpca.fit_transform(Xp_std)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
np.random.seed(4)
idx6 = np.random.choice(len(Xp_std), 3000, replace=False)

# Robust vs Classic Mahalanobis distance scatter
ax = axes[0]
not_out = ~outlier_mask
subsample_mask = not_out & (np.random.rand(len(Xp_std)) < 0.3)
ax.scatter(maha_classic[subsample_mask], maha_robust[subsample_mask],
           s=5, alpha=0.3, color='steelblue', label='Inlier')
ax.scatter(maha_classic[outlier_mask], maha_robust[outlier_mask],
           s=20, alpha=0.7, color='crimson', label=f'Outlier (n={n_outliers})', zorder=5)
ax.axhline(thresh_robust, color='crimson', ls='--', lw=1.5,
           label=f'Robust threshold ({thresh_robust:.2f})')
ax.set_xlabel('Classical Mahalanobis distance')
ax.set_ylabel('Robust (MCD) Mahalanobis distance')
ax.set_title('(A) Robust outlier detection\n(photometric features, n=10,000)')
ax.legend(fontsize=8)

# Robust PCA projection
ax = axes[1]
for cls in cls_names:
    m = y[idx6] == cls
    ax.scatter(Z_robust_pca[idx6][m, 0], Z_robust_pca[idx6][m, 1],
               s=8, alpha=0.45, color=COLORS[cls], label=cls)
ax.set_xlabel('Robust PC1'); ax.set_ylabel('Robust PC2')
ax.set_title('(B) Robust PCA projection\n(MCD covariance, n=3,000)')
ax.legend(markerscale=3)

# Kernel PCA (RBF)
ax = axes[2]
for cls in cls_names:
    m = y[idx6] == cls
    ax.scatter(Z_kpca[idx6][m, 0], Z_kpca[idx6][m, 1],
               s=8, alpha=0.45, color=COLORS[cls], label=cls)
ax.set_xlabel('Kernel PC1 (RBF)'); ax.set_ylabel('Kernel PC2 (RBF)')
ax.set_title('(C) Kernel PCA (RBF, γ=0.5)\n(photometric features, n=3,000)')
ax.legend(markerscale=3)

fig.suptitle('Figure 8. Extensions: Robust Estimation and Kernel PCA',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('/home/user/sta437finalprojecct/figures/fig8_robust_kernel.png')
plt.close()
print("Figure 8 saved.")

# ============================================================
# Summary stats for report
# ============================================================
print("\n=== SUMMARY FOR REPORT ===")
print(f"Dataset: {len(df)} objects  |  Features: {df.shape[1]}")
print(f"Class balance: {df['class'].value_counts().to_dict()}")
print(f"\nPCA (photo+redshift):")
print(f"  PC1 explains {pca.explained_variance_ratio_[0]*100:.1f}%  |  PC1+PC2: {pca.explained_variance_ratio_[:2].sum()*100:.1f}%")
print(f"\nGMM ARI:")
print(f"  Photo only: {ari_photo:.3f}  |  +redshift: {ari_full:.3f}")
print(f"\nClassification CV accuracy:")
print(f"  LDA photo:      {acc_lda_p.mean():.3f}")
print(f"  LDA +redshift:  {acc_lda_a.mean():.3f}")
print(f"  LR  photo:      {acc_lr_p.mean():.3f}")
print(f"  LR  +redshift:  {acc_lr_a.mean():.3f}")
print(f"\nRobust outliers: {n_outliers} ({n_outliers/len(df)*100:.1f}%)")

# Confusion matrix on LDA with photo only (full train)
lda_p_full = LinearDiscriminantAnalysis().fit(Xp, y)
preds_p = lda_p_full.predict(Xp)
print("\nLDA (photo only) full-data confusion matrix:")
print(pd.DataFrame(confusion_matrix(y, preds_p, labels=cls_names),
                   index=cls_names, columns=cls_names))

lda_a_full = LinearDiscriminantAnalysis().fit(Xa, y)
preds_a = lda_a_full.predict(Xa)
print("\nLDA (photo+redshift) full-data confusion matrix:")
print(pd.DataFrame(confusion_matrix(y, preds_a, labels=cls_names),
                   index=cls_names, columns=cls_names))
print("\nAll figures saved to /home/user/sta437finalprojecct/figures/")
