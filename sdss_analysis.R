# =============================================================
# STA437 Final Project — SDSS Stellar Classification (R)
# Central question: How well can photometric measurements alone
# recover spectroscopic classifications of celestial objects?
# =============================================================

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(MASS)
  library(mclust)
  library(kernlab)
  library(robustbase)
  library(corrplot)
  library(factoextra)
  library(FactoMineR)
  library(GGally)
  library(gridExtra)
  library(caret)
  library(grid)
})

set.seed(437)
dir.create("figures_r", showWarnings = FALSE)

# Colour palette (consistent with report)
CLS_COLORS <- c("GALAXY" = "#FF5722", "STAR" = "#2196F3", "QSO" = "#4CAF50")

# ============================================================
# 0. Data Loading
# ============================================================
df <- read.csv("sdss_data.csv", stringsAsFactors = FALSE)
df$class <- factor(df$class, levels = c("GALAXY", "STAR", "QSO"))
cat(sprintf("Loaded %d objects, %d variables\n", nrow(df), ncol(df)))
cat("Classes:\n"); print(table(df$class))

photo_cols   <- c("u", "g", "r", "i", "z")
all_phys     <- c(photo_cols, "redshift")

# Colour indices
df$u_g <- df$u - df$g
df$g_r <- df$g - df$r
df$r_i <- df$r - df$i
df$i_z <- df$i - df$z
color_cols <- c("u_g", "g_r", "r_i", "i_z")

cat("\nBasic statistics (photometric + redshift):\n")
print(summary(df[, all_phys]))

# ============================================================
# Mardia's test (manual implementation)
# ============================================================
mardia_test <- function(X) {
  X <- as.matrix(X)
  n <- nrow(X); p <- ncol(X)
  mu   <- colMeans(X)
  S    <- cov(X)
  Sinv <- solve(S + diag(1e-10, p))
  diff <- sweep(X, 2, mu)
  D2   <- rowSums((diff %*% Sinv) * diff)
  # Skewness
  G <- diff %*% Sinv %*% t(diff)
  b1p <- mean(G^3)
  k_skew <- n * b1p / 6
  df_skew <- p * (p + 1) * (p + 2) / 6
  p_skew  <- 1 - pchisq(k_skew, df = df_skew)
  # Kurtosis
  b2p   <- mean(D2^2)
  mu_k  <- p * (p + 2)
  sig_k <- sqrt(8 * p * (p + 2) / n)
  z_k   <- (b2p - mu_k) / sig_k
  p_kurt <- 2 * (1 - pnorm(abs(z_k)))
  list(D2 = D2, k_skew = k_skew, p_skew = p_skew,
       b2p = b2p, z_kurt = z_k, p_kurt = p_kurt)
}

# ============================================================
# FIGURE 1 — EDA: Class distribution and magnitude histograms
# ============================================================
cat("\nGenerating Figure 1...\n")

# Panel A: class counts
p_counts <- ggplot(df, aes(x = class, fill = class)) +
  geom_bar(color = "white", width = 0.6) +
  geom_text(stat = "count",
            aes(label = paste0(after_stat(count), "\n(",
                               round(after_stat(count)/nrow(df)*100), "%)")),
            vjust = -0.3, size = 3.2) +
  scale_fill_manual(values = CLS_COLORS) +
  scale_y_continuous(limits = c(0, 6200)) +
  labs(title = "(A) Class distribution", x = NULL, y = "Count") +
  theme_bw(base_size = 10) + theme(legend.position = "none")

# Panel B: redshift distribution
p_rs <- df %>%
  mutate(redshift_clip = pmax(redshift, 0)) %>%
  ggplot(aes(x = redshift_clip, fill = class, color = class)) +
  geom_density(alpha = 0.55, bw = 0.015) +
  scale_fill_manual(values = CLS_COLORS) +
  scale_color_manual(values = CLS_COLORS) +
  coord_cartesian(xlim = c(-0.02, 0.8)) +
  labs(title = "(B) Redshift distributions by class",
       x = "Redshift (clipped at 0)", y = "Density", fill = NULL, color = NULL) +
  theme_bw(base_size = 10) + theme(legend.position = "bottom")

# Panels C-G: per-band histograms
band_plots <- lapply(seq_along(photo_cols), function(k) {
  col  <- photo_cols[k]
  sub  <- df %>% filter(.data[[col]] > 10, .data[[col]] < 26)
  lbl  <- paste0("(", LETTERS[k + 2], ") ", col, " band")
  ggplot(sub, aes(x = .data[[col]], fill = class)) +
    geom_histogram(aes(y = after_stat(density)), bins = 40,
                   alpha = 0.6, position = "identity", color = NA) +
    scale_fill_manual(values = CLS_COLORS) +
    labs(title = lbl, x = paste0(col, " (mag)"), y = "Density",
         fill = NULL) +
    theme_bw(base_size = 10) +
    theme(legend.position = if (k == 1) "bottom" else "none")
})

fig1 <- grid.arrange(
  arrangeGrob(p_counts, p_rs, ncol = 2),
  arrangeGrob(grobs = band_plots, ncol = 3),
  nrow = 2,
  top = textGrob(
    "Figure 1. SDSS Dataset Overview: Class Distribution and Photometric Magnitudes",
    gp = gpar(fontsize = 11, fontface = "bold"))
)
ggsave("figures_r/fig1_eda_overview.png", fig1, width = 14, height = 10, dpi = 150)
cat("  Figure 1 saved.\n")

# ============================================================
# FIGURE 2 — Correlation matrix and colour-colour diagram
# ============================================================
cat("Generating Figure 2...\n")

feat_sub <- df[, c(all_phys, color_cols)]
colnames(feat_sub) <- c("u","g","r","i","z","redshift","u-g","g-r","r-i","i-z")
corr_mat <- cor(feat_sub)

png("figures_r/fig2_correlations.png", width = 1400, height = 550, res = 150)
par(mfrow = c(1, 2), mar = c(2, 2, 3, 2))
corrplot(corr_mat, method = "color", type = "full", addCoef.col = "black",
         number.cex = 0.65, tl.cex = 0.75, cl.cex = 0.7, tl.col = "black",
         col = colorRampPalette(c("#053061","#FFFFFF","#67001F"))(200),
         title = "(A) Feature correlation matrix", mar = c(0,0,2,0))

idx_sub <- sample(nrow(df), 3000)
sub_cc  <- df[idx_sub, ]
# clip extreme colours for display
sub_cc  <- sub_cc %>% filter(u_g > -1, u_g < 5, g_r > -1, g_r < 3)
plot(sub_cc$u_g, sub_cc$g_r,
     col = adjustcolor(CLS_COLORS[as.character(sub_cc$class)], alpha.f = 0.4),
     pch = 16, cex = 0.4,
     xlab = "u - g (mag)", ylab = "g - r (mag)",
     main = "(B) Colour-colour diagram (u-g vs g-r)\n3,000 random objects",
     cex.main = 0.9)
legend("topright", legend = names(CLS_COLORS),
       col = CLS_COLORS, pch = 16, cex = 0.8)
dev.off()
cat("  Figure 2 saved.\n")

# ============================================================
# FIGURE 3 — MVN Assessment (Mahalanobis QQ plots)
# ============================================================
cat("Generating Figure 3...\n")

X_photo <- scale(as.matrix(df[, photo_cols]))
p_dim   <- ncol(X_photo)
n_tot   <- nrow(X_photo)

mt_full <- mardia_test(X_photo)
cat(sprintf("  Mardia full: skew stat=%.1f p=%.2e  kurt z=%.1f p=%.2e\n",
            mt_full$k_skew, mt_full$p_skew, mt_full$z_kurt, mt_full$p_kurt))
for (cls in levels(df$class)) {
  Xc <- X_photo[df$class == cls, ]
  mt <- mardia_test(Xc)
  cat(sprintf("  %s: skew p=%.2e  kurt p=%.2e\n", cls, mt$p_skew, mt$p_kurt))
}

png("figures_r/fig3_mvn_qqplot.png", width = 1500, height = 420, res = 150)
par(mfrow = c(1, 4), mar = c(4, 4, 3, 1))

# Full data
chi2q_full <- qchisq(ppoints(n_tot), df = p_dim)
plot(sort(chi2q_full), sort(mt_full$D2),
     pch = ".", cex = 1.5, col = adjustcolor("steelblue", 0.5),
     xlab = expression(chi^2*(5)~quantiles), ylab = "Mahalanobis D²",
     main = "All objects (n=10,000)", cex.main = 0.9)
abline(0, 1, col = "red", lwd = 1.5, lty = 2)

# Per-class
cls_colors_vec <- c("GALAXY" = "#FF5722", "STAR" = "#2196F3", "QSO" = "#4CAF50")
for (cls in levels(df$class)) {
  Xc  <- X_photo[df$class == cls, ]
  nc  <- nrow(Xc)
  mt  <- mardia_test(Xc)
  chi2q <- qchisq(ppoints(nc), df = p_dim)
  plot(sort(chi2q), sort(mt$D2),
       pch = ".", cex = 1.5, col = adjustcolor(cls_colors_vec[cls], 0.5),
       xlab = expression(chi^2*(5)~quantiles), ylab = "Mahalanobis D²",
       main = paste0(cls, " (n=", nc, ")"), cex.main = 0.9)
  abline(0, 1, col = "red", lwd = 1.5, lty = 2)
}
dev.off()
cat("  Figure 3 saved.\n")

# ============================================================
# FIGURE 4 — PCA
# ============================================================
cat("Generating Figure 4...\n")

X_full_std <- scale(as.matrix(df[, all_phys]))
pca_res    <- prcomp(X_full_std, center = FALSE, scale. = FALSE)
var_exp    <- (pca_res$sdev^2) / sum(pca_res$sdev^2)
cat("  PCA variance explained:\n")
for (i in seq_along(var_exp)) {
  cat(sprintf("    PC%d: %.3f  cumulative: %.3f\n", i, var_exp[i], cumsum(var_exp)[i]))
}

scores_pca <- as.data.frame(pca_res$x)
scores_pca$class <- df$class
idx2 <- sample(nrow(scores_pca), 3000)

p_scree <- data.frame(PC = 1:6, var = var_exp * 100,
                       cum = cumsum(var_exp) * 100) %>%
  ggplot() +
  geom_col(aes(x = PC, y = var), fill = "steelblue", alpha = 0.8, width = 0.6) +
  geom_line(aes(x = PC, y = cum), color = "darkorange", linewidth = 1.2) +
  geom_point(aes(x = PC, y = cum), color = "darkorange", size = 2.5) +
  geom_hline(yintercept = 90, linetype = "dashed", color = "grey50", linewidth = 0.8) +
  annotate("text", x = 5.5, y = 91.5, label = "90%", size = 3, color = "grey40") +
  labs(title = "(A) Scree plot",
       x = "Principal component", y = "Explained variance (%)") +
  scale_x_continuous(breaks = 1:6) +
  theme_bw(base_size = 10)

p_scores <- ggplot(scores_pca[idx2, ],
                   aes(x = PC1, y = PC2, color = class)) +
  geom_point(size = 0.8, alpha = 0.5) +
  scale_color_manual(values = CLS_COLORS) +
  labs(title = sprintf("(B) PC1 vs PC2 (n=3,000)"),
       x = sprintf("PC1 (%.1f%%)", var_exp[1]*100),
       y = sprintf("PC2 (%.1f%%)", var_exp[2]*100),
       color = NULL) +
  theme_bw(base_size = 10) + theme(legend.position = "bottom")

# Loadings heatmap
loadings_df <- as.data.frame(pca_res$rotation[, 1:6])
loadings_df$feature <- rownames(loadings_df)
loadings_long <- pivot_longer(loadings_df, -feature,
                               names_to = "PC", values_to = "loading")
p_loadings <- ggplot(loadings_long, aes(x = PC, y = feature, fill = loading)) +
  geom_tile(color = "white") +
  geom_text(aes(label = sprintf("%.2f", loading)), size = 3.2) +
  scale_fill_gradient2(low = "#053061", mid = "white", high = "#67001F",
                       midpoint = 0, limits = c(-1, 1)) +
  labs(title = "(C) PCA loadings", x = NULL, y = NULL, fill = "Loading") +
  theme_bw(base_size = 10)

fig4 <- arrangeGrob(p_scree, p_scores, p_loadings, ncol = 3,
  top = textGrob(
    "Figure 4. Principal Component Analysis (photometric magnitudes + redshift)",
    gp = gpar(fontsize = 11, fontface = "bold")))
ggsave("figures_r/fig4_pca.png", fig4, width = 15, height = 5, dpi = 150)
cat("  Figure 4 saved.\n")

# ============================================================
# FIGURE 5 — Factor Analysis
# ============================================================
cat("Generating Figure 5...\n")

X_photo_std <- scale(as.matrix(df[, photo_cols]))

# Log-likelihood by n_factors (max 2 for 5 variables with factanal)
ll_vals <- sapply(1:2, function(nf) {
  fa <- factanal(X_photo_std, factors = nf, rotation = "varimax", nstart = 5)
  logLik_val <- -0.5 * (fa$STATISTIC + fa$dof * log(2 * pi))
  logLik_val / nrow(X_photo_std)
})

fa2 <- factanal(X_photo_std, factors = 2, rotation = "varimax",
                scores = "regression", nstart = 5)
cat("  Factor loadings (2-factor varimax):\n")
print(loadings(fa2), cutoff = 0)

fa_scores_df <- as.data.frame(fa2$scores)
colnames(fa_scores_df) <- c("Factor1", "Factor2")
fa_scores_df$class <- df$class

p_fa_ll <- data.frame(nf = 1:2, ll = ll_vals) %>%
  ggplot(aes(x = nf, y = ll)) +
  geom_line(color = "steelblue", linewidth = 1.2) +
  geom_point(color = "steelblue", size = 3) +
  labs(title = "(A) Model fit vs. number of factors",
       x = "Number of factors", y = "Log-likelihood per sample") +
  scale_x_continuous(breaks = 1:2) +
  theme_bw(base_size = 10)

# Loadings heatmap
load_mat <- as.matrix(fa2$loadings)
load_df  <- data.frame(feature = rownames(load_mat),
                        Factor1 = load_mat[,1], Factor2 = load_mat[,2])
load_long <- pivot_longer(load_df, -feature, names_to = "Factor", values_to = "loading")
p_fa_load <- ggplot(load_long, aes(x = Factor, y = feature, fill = loading)) +
  geom_tile(color = "white") +
  geom_text(aes(label = sprintf("%.2f", loading)), size = 4) +
  scale_fill_gradient2(low = "#053061", mid = "white", high = "#67001F",
                       midpoint = 0, limits = c(-1, 1)) +
  labs(title = "(B) Factor loadings (varimax)\n2-factor model",
       x = NULL, y = NULL, fill = "Loading") +
  theme_bw(base_size = 10)

idx3 <- sample(nrow(fa_scores_df), 3000)
p_fa_scores <- ggplot(fa_scores_df[idx3, ],
                       aes(x = Factor1, y = Factor2, color = class)) +
  geom_point(size = 0.8, alpha = 0.5) +
  scale_color_manual(values = CLS_COLORS) +
  labs(title = "(C) Factor scores by class (n=3,000)",
       x = "Factor 1 score", y = "Factor 2 score", color = NULL) +
  theme_bw(base_size = 10) + theme(legend.position = "bottom")

fig5 <- arrangeGrob(p_fa_ll, p_fa_load, p_fa_scores, ncol = 3,
  top = textGrob(
    "Figure 5. Factor Analysis of Photometric Magnitudes (varimax rotation)",
    gp = gpar(fontsize = 11, fontface = "bold")))
ggsave("figures_r/fig5_factor_analysis.png", fig5, width = 15, height = 5, dpi = 150)
cat("  Figure 5 saved.\n")

# ============================================================
# FIGURE 6 — GMM Clustering
# ============================================================
cat("Generating Figure 6...\n")

X_photo_s  <- scale(as.matrix(df[, photo_cols]))
X_full_s   <- scale(as.matrix(df[, all_phys]))

# PCA for 2-D projection
Z_photo_2d <- prcomp(X_photo_s, center = FALSE)$x[, 1:2]
Z_full_2d  <- prcomp(X_full_s,  center = FALSE)$x[, 1:2]

# GMM with 3 components (mclust selects covariance type via BIC)
gmm_photo <- Mclust(X_photo_s, G = 3, verbose = FALSE)
gmm_full  <- Mclust(X_full_s,  G = 3, verbose = FALSE)

# ARI
true_num <- as.integer(df$class)  # 1=GALAXY, 2=STAR, 3=QSO
ari_photo <- adjustedRandIndex(gmm_photo$classification, true_num)
ari_full  <- adjustedRandIndex(gmm_full$classification,  true_num)
cat(sprintf("  GMM ARI — photo only: %.3f  |  photo+redshift: %.3f\n",
            ari_photo, ari_full))

idx4 <- sample(nrow(df), 3000)
pca_plot_df <- data.frame(
  PC1 = Z_photo_2d[idx4, 1], PC2 = Z_photo_2d[idx4, 2],
  true_class = df$class[idx4],
  gmm_photo  = factor(gmm_photo$classification[idx4]),
  PC1f = Z_full_2d[idx4, 1], PC2f = Z_full_2d[idx4, 2],
  gmm_full   = factor(gmm_full$classification[idx4])
)

p_true <- ggplot(pca_plot_df, aes(x = PC1, y = PC2, color = true_class)) +
  geom_point(size = 0.6, alpha = 0.4) +
  scale_color_manual(values = CLS_COLORS, name = NULL) +
  labs(title = "(A) True labels (photo PCA space)",
       x = "PC1", y = "PC2") +
  theme_bw(base_size = 10) + theme(legend.position = "bottom")

p_gmm_p <- ggplot(pca_plot_df, aes(x = PC1, y = PC2, color = gmm_photo)) +
  geom_point(size = 0.6, alpha = 0.4) +
  scale_color_manual(values = c("1"="#FF5722","2"="#2196F3","3"="#4CAF50")) +
  labs(title = sprintf("(B) GMM — photo only\nARI = %.3f", ari_photo),
       x = "PC1", y = "PC2", color = "Cluster") +
  theme_bw(base_size = 10) + theme(legend.position = "bottom")

p_gmm_f <- ggplot(pca_plot_df, aes(x = PC1f, y = PC2f, color = gmm_full)) +
  geom_point(size = 0.6, alpha = 0.4) +
  scale_color_manual(values = c("1"="#FF5722","2"="#2196F3","3"="#4CAF50")) +
  labs(title = sprintf("(C) GMM — photo + redshift\nARI = %.3f", ari_full),
       x = "PC1 (with redshift)", y = "PC2", color = "Cluster") +
  theme_bw(base_size = 10) + theme(legend.position = "bottom")

fig6 <- arrangeGrob(p_true, p_gmm_p, p_gmm_f, ncol = 3,
  top = textGrob(
    "Figure 6. GMM Clustering: Photometry Alone vs. Photometry + Redshift",
    gp = gpar(fontsize = 11, fontface = "bold")))
ggsave("figures_r/fig6_gmm_clustering.png", fig6, width = 15, height = 5.5, dpi = 150)
cat("  Figure 6 saved.\n")

# ============================================================
# FIGURE 7 — LDA + 5-fold cross-validation
# ============================================================
cat("Generating Figure 7...\n")

Xp <- as.data.frame(scale(df[, photo_cols]));  Xp$class <- df$class
Xa <- as.data.frame(scale(df[, all_phys]));    Xa$class <- df$class

lda_p <- lda(class ~ ., data = Xp)
lda_a <- lda(class ~ ., data = Xa)
Z_lda_p <- predict(lda_p)$x
Z_lda_a <- predict(lda_a)$x

# 5-fold CV
cv_folds <- createFolds(df$class, k = 5, returnTrain = TRUE)
cv_acc <- function(X_df) {
  sapply(cv_folds, function(train_idx) {
    tr <- X_df[train_idx, ]; te <- X_df[-train_idx, ]
    mod <- lda(class ~ ., data = tr)
    preds <- predict(mod, te)$class
    mean(preds == te$class)
  })
}
# Also LR (multinomial via nnet via caret)
cv_lr <- function(X_df) {
  sapply(cv_folds, function(train_idx) {
    tr <- X_df[train_idx, ]; te <- X_df[-train_idx, ]
    mod <- nnet::multinom(class ~ ., data = tr, trace = FALSE, MaxNWts = 5000)
    preds <- predict(mod, te)
    mean(preds == te$class)
  })
}

acc_lda_p <- cv_acc(Xp); acc_lda_a <- cv_acc(Xa)
acc_lr_p  <- cv_lr(Xp);  acc_lr_a  <- cv_lr(Xa)

cat(sprintf("  LDA photo:     %.3f ± %.3f\n", mean(acc_lda_p), sd(acc_lda_p)))
cat(sprintf("  LDA +redshift: %.3f ± %.3f\n", mean(acc_lda_a), sd(acc_lda_a)))
cat(sprintf("  LR  photo:     %.3f ± %.3f\n", mean(acc_lr_p),  sd(acc_lr_p)))
cat(sprintf("  LR  +redshift: %.3f ± %.3f\n", mean(acc_lr_a),  sd(acc_lr_a)))

idx5 <- sample(nrow(df), 3000)
lda_plot_p <- data.frame(LD1 = Z_lda_p[idx5,1], LD2 = Z_lda_p[idx5,2], class = df$class[idx5])
lda_plot_a <- data.frame(LD1 = Z_lda_a[idx5,1], LD2 = Z_lda_a[idx5,2], class = df$class[idx5])

p_lda_p <- ggplot(lda_plot_p, aes(x = LD1, y = LD2, color = class)) +
  geom_point(size = 0.7, alpha = 0.45) +
  scale_color_manual(values = CLS_COLORS, name = NULL) +
  labs(title = sprintf("(A) LDA — photo only\nCV acc = %.3f", mean(acc_lda_p)),
       x = "LD1", y = "LD2") +
  theme_bw(base_size = 10) + theme(legend.position = "bottom")

p_lda_a <- ggplot(lda_plot_a, aes(x = LD1, y = LD2, color = class)) +
  geom_point(size = 0.7, alpha = 0.45) +
  scale_color_manual(values = CLS_COLORS, name = NULL) +
  labs(title = sprintf("(B) LDA — photo + redshift\nCV acc = %.3f", mean(acc_lda_a)),
       x = "LD1", y = "LD2") +
  theme_bw(base_size = 10) + theme(legend.position = "bottom")

acc_df <- data.frame(
  method = factor(c("LDA\n(photo)","LDA\n(photo+z)","LR\n(photo)","LR\n(photo+z)"),
                  levels = c("LDA\n(photo)","LDA\n(photo+z)","LR\n(photo)","LR\n(photo+z)")),
  mean   = c(mean(acc_lda_p), mean(acc_lda_a), mean(acc_lr_p), mean(acc_lr_a)),
  sd     = c(sd(acc_lda_p),   sd(acc_lda_a),   sd(acc_lr_p),  sd(acc_lr_a)),
  fill   = c("#90CAF9","#1565C0","#FFAB91","#BF360C")
)
p_cv <- ggplot(acc_df, aes(x = method, y = mean, fill = fill)) +
  geom_col(width = 0.5, color = "white") +
  geom_errorbar(aes(ymin = mean - sd, ymax = mean + sd), width = 0.2) +
  geom_text(aes(y = mean + sd + 0.01, label = sprintf("%.3f", mean)),
            size = 3.2, vjust = 0) +
  scale_fill_identity() +
  scale_y_continuous(limits = c(0.5, 1.06)) +
  labs(title = "(C) Classification accuracy\n(error bars = ±1 SD across folds)",
       x = NULL, y = "5-fold CV accuracy") +
  theme_bw(base_size = 10)

fig7 <- arrangeGrob(p_lda_p, p_lda_a, p_cv, ncol = 3,
  top = textGrob(
    "Figure 7. Linear Discriminant Analysis and Cross-Validated Classification Accuracy",
    gp = gpar(fontsize = 11, fontface = "bold")))
ggsave("figures_r/fig7_lda_cv.png", fig7, width = 15, height = 5.5, dpi = 150)
cat("  Figure 7 saved.\n")

# ============================================================
# FIGURE 8 — Robust PCA (MCD) + Kernel PCA
# ============================================================
cat("Generating Figure 8...\n")

X_ps <- scale(as.matrix(df[, photo_cols]))

# Robust covariance (MCD, support fraction 0.85)
mcd_fit    <- covMcd(X_ps, alpha = 0.85)
mu_robust  <- mcd_fit$center
S_robust   <- mcd_fit$cov

# Classical Mahalanobis
mu_c <- colMeans(X_ps)
S_c  <- cov(X_ps)
maha_c <- mahalanobis(X_ps, mu_c, S_c)
maha_r <- mahalanobis(X_ps, mu_robust, S_robust)

thresh_r  <- sqrt(qchisq(0.975, df = length(photo_cols)))
out_mask  <- sqrt(maha_r) > thresh_r
n_out     <- sum(out_mask)
cat(sprintf("  Robust outliers: %d (%.1f%%)\n", n_out, n_out/nrow(X_ps)*100))
cat("  Outlier class breakdown:\n"); print(table(df$class[out_mask]))

# Robust PCA
eig_res     <- eigen(S_robust)
order_e     <- order(eig_res$values, decreasing = TRUE)
evecs       <- eig_res$vectors[, order_e[1:2]]
Z_rpca      <- sweep(X_ps, 2, mu_robust) %*% evecs
colnames(Z_rpca) <- c("RPC1","RPC2")

# Kernel PCA (RBF, sigma=0.5) — use 3000 random points for speed
set.seed(99)
kpca_idx  <- sample(nrow(X_ps), 3000)
kpca_res  <- kpca(X_ps[kpca_idx, ], kernel = "rbfdot",
                   kpar = list(sigma = 0.5), features = 2)
Z_kpca    <- rotated(kpca_res)
colnames(Z_kpca) <- c("KPC1","KPC2")

idx6     <- sample(nrow(X_ps), 3000)
not_out  <- !out_mask
sub_inl  <- sample(which(not_out), min(1500, sum(not_out)))

out_plot_df <- data.frame(
  maha_c = sqrt(maha_c), maha_r = sqrt(maha_r),
  outlier = out_mask, class = df$class)
p_rob <- ggplot() +
  geom_point(data = out_plot_df[sub_inl,],
             aes(x = maha_c, y = maha_r), color = "steelblue",
             size = 0.6, alpha = 0.3) +
  geom_point(data = out_plot_df[out_mask,],
             aes(x = maha_c, y = maha_r), color = "#DC143C",
             size = 1.5, alpha = 0.7) +
  geom_hline(yintercept = thresh_r, color = "#DC143C",
             linetype = "dashed", linewidth = 0.8) +
  annotate("text", x = max(out_plot_df$maha_c)*0.6, y = thresh_r + 0.15,
           label = sprintf("Robust threshold (%.2f)", thresh_r),
           color = "#DC143C", size = 2.8) +
  labs(title = sprintf("(A) Robust outlier detection\n(n outliers=%d, %.1f%%)",
                        n_out, n_out/nrow(X_ps)*100),
       x = "Classical Mahalanobis dist.",
       y = "Robust (MCD) Mahalanobis dist.") +
  theme_bw(base_size = 10)

rpca_df <- as.data.frame(Z_rpca[idx6, ])
rpca_df$class <- df$class[idx6]
p_rpca <- ggplot(rpca_df, aes(x = RPC1, y = RPC2, color = class)) +
  geom_point(size = 0.7, alpha = 0.45) +
  scale_color_manual(values = CLS_COLORS, name = NULL) +
  labs(title = "(B) Robust PCA projection (MCD)",
       x = "Robust PC1", y = "Robust PC2") +
  theme_bw(base_size = 10) + theme(legend.position = "bottom")

kpca_df <- as.data.frame(Z_kpca)
colnames(kpca_df) <- c("KPC1", "KPC2")
kpca_df$class <- df$class[kpca_idx]
p_kpca <- ggplot(kpca_df, aes(x = KPC1, y = KPC2, color = class)) +
  geom_point(size = 0.7, alpha = 0.45) +
  scale_color_manual(values = CLS_COLORS, name = NULL) +
  labs(title = "(C) Kernel PCA (RBF, σ=0.5)",
       x = "Kernel PC1", y = "Kernel PC2") +
  theme_bw(base_size = 10) + theme(legend.position = "bottom")

fig8 <- arrangeGrob(p_rob, p_rpca, p_kpca, ncol = 3,
  top = textGrob(
    "Figure 8. Extensions: Robust Estimation (MCD) and Kernel PCA",
    gp = gpar(fontsize = 11, fontface = "bold")))
ggsave("figures_r/fig8_robust_kernel.png", fig8, width = 15, height = 5.5, dpi = 150)
cat("  Figure 8 saved.\n")

# ============================================================
# Summary statistics
# ============================================================
cat("\n========== SUMMARY FOR REPORT ==========\n")
cat(sprintf("PCA: PC1=%.1f%%  PC1+PC2=%.1f%%\n",
            var_exp[1]*100, sum(var_exp[1:2])*100))
cat(sprintf("GMM ARI: photo=%.3f  +redshift=%.3f\n", ari_photo, ari_full))
cat(sprintf("CV acc: LDA_p=%.3f  LDA_a=%.3f  LR_p=%.3f  LR_a=%.3f\n",
            mean(acc_lda_p), mean(acc_lda_a), mean(acc_lr_p), mean(acc_lr_a)))
cat(sprintf("Robust outliers: %d (%.1f%%)  QSO: %d/850\n",
            n_out, n_out/nrow(df)*100,
            sum(out_mask & df$class=="QSO")))
cat("\nAll R figures saved to figures_r/\n")
